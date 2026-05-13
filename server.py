#!/usr/bin/env python3
"""
Chrome Web Server — FastAPI + Playwright
Endpoints: /query  /fetch  /mcp
"""

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from playwright.async_api import Browser, async_playwright

load_dotenv()

MCP_API_KEY = os.getenv("MCP_API_KEY")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_pw = None
_browser: Browser = None
_mcp_page = None          # persistent page for /mcp sessions
_mcp_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pw, _browser, _mcp_page
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    )
    ctx = await _browser.new_context(user_agent=_UA)
    _mcp_page = await ctx.new_page()
    yield
    await _browser.close()
    await _pw.stop()


app = FastAPI(title="Chrome MCP Server", lifespan=lifespan)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ephemeral():
    """New throwaway (ctx, page). Caller must await ctx.close()."""
    ctx = await _browser.new_context(user_agent=_UA)
    page = await ctx.new_page()
    return ctx, page


def _require_auth(authorization: Optional[str]):
    if not MCP_API_KEY:
        raise HTTPException(500, "MCP_API_KEY env var not set")
    if authorization != f"Bearer {MCP_API_KEY}":
        raise HTTPException(401, "Unauthorized")


_GOOGLE_JS = r"""
() => {
    const out = [], seen = new Set();
    document.querySelectorAll('a[href]').forEach(a => {
        const h = a.href;
        if (!h.startsWith('http') || h.includes('google.com') || seen.has(h)) return;
        const h3 = a.querySelector('h3');
        if (!h3) return;
        let desc = '';
        const block = a.closest('div[data-hveid], div.g, li');
        if (block) {
            for (const el of block.querySelectorAll('span,div')) {
                const t = el.innerText?.trim() || '';
                if (t.length > 50 && t !== h3.innerText.trim()) { desc = t.slice(0, 300); break; }
            }
        }
        seen.add(h);
        out.push({ url: h, title: h3.innerText.trim(), description: desc });
    });
    return out;
}
"""

_CLEAN_TEXT_JS = r"""
() => {
    ['script','style','nav','footer','noscript','iframe','aside']
        .forEach(t => document.querySelectorAll(t).forEach(e => e.remove()));
    return document.body?.innerText?.trim() || '';
}
"""


# ---------------------------------------------------------------------------
# /query  — Google search → links + descriptions
# ---------------------------------------------------------------------------

@app.get("/query")
async def query_endpoint(q: str = Query(..., description="Search query string")):
    """
    Search Google for `q`, return all result links with title + description.
    """
    ctx, page = await _ephemeral()
    try:
        await page.goto(
            f"https://www.google.com/search?q={q}&hl=en&num=20",
            wait_until="domcontentloaded",
            timeout=15_000,
        )
        results = await page.evaluate(_GOOGLE_JS)
        return JSONResponse(content=results)
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# /fetch  — Visit URL → clean text content
# ---------------------------------------------------------------------------

@app.get("/fetch")
async def fetch_endpoint(url: str = Query(..., description="URL to visit")):
    """
    Visit `url`, strip boilerplate, return clean text + title + final URL.
    """
    if not url.startswith("http"):
        url = "https://" + url
    ctx, page = await _ephemeral()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        content = await page.evaluate(_CLEAN_TEXT_JS)
        return JSONResponse(
            content={"url": page.url, "title": await page.title(), "content": content}
        )
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# /mcp  — MCP (Model Context Protocol) over Streamable HTTP
# Requires:  Authorization: Bearer <MCP_API_KEY>
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "navigate",
        "description": "Navigate to a URL",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL"}},
            "required": ["url"],
        },
    },
    {
        "name": "get_content",
        "description": "Get cleaned text content of the current page",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_html",
        "description": "Get raw HTML of the current page",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "click",
        "description": "Click an element by CSS selector",
        "inputSchema": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
    {
        "name": "type_text",
        "description": "Fill a text input identified by CSS selector",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["selector", "text"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key (e.g. Enter, Tab, Escape)",
        "inputSchema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "screenshot",
        "description": "Take a screenshot, returns base64 PNG",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "evaluate",
        "description": "Execute JavaScript in the page and return the result",
        "inputSchema": {
            "type": "object",
            "properties": {"script": {"type": "string"}},
            "required": ["script"],
        },
    },
    {
        "name": "get_links",
        "description": "Get all href links on the current page",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "search",
        "description": "Google search — returns list of {url, title, description}",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page. direction: up | down | top | bottom",
        "inputSchema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "top", "bottom"],
                },
                "amount": {
                    "type": "integer",
                    "description": "Pixels (for up/down, default 600)",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for a CSS selector to appear",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "timeout_ms": {"type": "integer", "description": "Max ms (default 5000)"},
            },
            "required": ["selector"],
        },
    },
    {
        "name": "page_info",
        "description": "Return current URL and page title",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


async def _execute_tool(name: str, args: dict) -> Any:
    p = _mcp_page

    if name == "navigate":
        await p.goto(args["url"], wait_until="domcontentloaded", timeout=20_000)
        return {"url": p.url, "title": await p.title()}

    elif name == "get_content":
        text = await p.evaluate(_CLEAN_TEXT_JS)
        return {"url": p.url, "title": await p.title(), "content": text}

    elif name == "get_html":
        return {"html": await p.content()}

    elif name == "click":
        await p.click(args["selector"], timeout=5_000)
        await p.wait_for_load_state("domcontentloaded")
        return {"clicked": args["selector"], "url": p.url}

    elif name == "type_text":
        await p.fill(args["selector"], args["text"])
        return {"selector": args["selector"], "text": args["text"]}

    elif name == "press_key":
        await p.keyboard.press(args["key"])
        return {"key": args["key"]}

    elif name == "screenshot":
        data = await p.screenshot(type="png")
        return {"image": base64.b64encode(data).decode(), "encoding": "base64", "format": "png"}

    elif name == "evaluate":
        result = await p.evaluate(args["script"])
        return {"result": result}

    elif name == "get_links":
        links = await p.evaluate(
            "() => Array.from(document.querySelectorAll('a[href]'))"
            ".map(a => ({ url: a.href, text: a.innerText.trim().slice(0,200) }))"
            ".filter(l => l.url.startsWith('http'))"
        )
        return {"links": links}

    elif name == "search":
        await p.goto(
            f"https://www.google.com/search?q={args['query']}&hl=en",
            wait_until="domcontentloaded",
        )
        results = await p.evaluate(_GOOGLE_JS)
        return {"results": results}

    elif name == "scroll":
        d, amt = args["direction"], args.get("amount", 600)
        if d == "down":
            await p.evaluate(f"window.scrollBy(0, {amt})")
        elif d == "up":
            await p.evaluate(f"window.scrollBy(0, -{amt})")
        elif d == "top":
            await p.evaluate("window.scrollTo(0,0)")
        elif d == "bottom":
            await p.evaluate("window.scrollTo(0,document.body.scrollHeight)")
        return {"scrolled": d, "amount": amt}

    elif name == "wait":
        await p.wait_for_selector(args["selector"], timeout=args.get("timeout_ms", 5_000))
        return {"found": args["selector"]}

    elif name == "page_info":
        return {"url": p.url, "title": await p.title()}

    else:
        raise ValueError(f"Unknown tool: {name}")


@app.post("/mcp")
async def mcp_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    MCP Streamable HTTP transport (JSON-RPC 2.0).
    All calls require:  Authorization: Bearer <MCP_API_KEY>
    """
    _require_auth(authorization)

    body = await request.json()
    method = body.get("method", "")
    params = body.get("params", {})
    req_id = body.get("id")

    def _ok(result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _err(code, msg):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    # ---- dispatch ----
    if method == "initialize":
        payload = _ok(
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "chrome-mcp", "version": "1.0.0"},
            }
        )

    elif method in ("notifications/initialized", "ping"):
        # fire-and-forget or no-op
        payload = None

    elif method == "tools/list":
        payload = _ok({"tools": MCP_TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        async with _mcp_lock:
            try:
                result = await _execute_tool(tool_name, tool_args)
                payload = _ok(
                    {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}
                )
            except Exception as exc:
                payload = _ok(
                    {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    }
                )

    else:
        payload = _err(-32601, f"Method not found: {method}")

    if payload is None:
        return JSONResponse(content={}, status_code=204)

    # Honour SSE Accept header (some MCP clients request it)
    if "text/event-stream" in request.headers.get("accept", ""):
        async def _sse():
            yield f"data: {json.dumps(payload)}\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")

    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
