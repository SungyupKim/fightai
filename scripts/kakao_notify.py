"""Sends a KakaoTalk "나에게 보내기" (send-to-me) notification -- used to ping when a
training round finishes. Token file lives in checkpoints/ (already gitignored), never
committed. REST API key is passed in, never hardcoded here.

One-time setup:
    python kakao_notify.py auth --rest-api-key KEY --redirect-uri URI --code CODE
      -- exchanges the OAuth code for access/refresh tokens, saves them locally.

Then just:
    python kakao_notify.py send "message text"
      -- sends a message, auto-refreshing the access token first if it's expired.

Or from another script:
    from kakao_notify import send_message
    send_message("round 5 done")
"""
import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

TOKEN_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints" / "kakao_token.json"


def _post(url, data):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def exchange_code(rest_api_key, redirect_uri, code, client_secret=None):
    data = {
        "grant_type": "authorization_code",
        "client_id": rest_api_key,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    if client_secret:
        data["client_secret"] = client_secret
    result = _post("https://kauth.kakao.com/oauth/token", data)
    result["rest_api_key"] = rest_api_key
    if client_secret:
        result["client_secret"] = client_secret
    result["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(result))
    print(f"saved tokens to {TOKEN_PATH}")
    return result


def _load_tokens():
    if not TOKEN_PATH.exists():
        raise RuntimeError(f"no token file at {TOKEN_PATH} -- run the 'auth' subcommand first")
    return json.loads(TOKEN_PATH.read_text())


def _refresh(tokens):
    data = {
        "grant_type": "refresh_token",
        "client_id": tokens["rest_api_key"],
        "refresh_token": tokens["refresh_token"],
    }
    if tokens.get("client_secret"):
        data["client_secret"] = tokens["client_secret"]
    result = _post("https://kauth.kakao.com/oauth/token", data)
    tokens["access_token"] = result["access_token"]
    tokens["expires_in"] = result.get("expires_in", tokens.get("expires_in"))
    if "refresh_token" in result:  # Kakao sometimes rotates the refresh token too
        tokens["refresh_token"] = result["refresh_token"]
    tokens["obtained_at"] = time.time()
    TOKEN_PATH.write_text(json.dumps(tokens))
    return tokens


def send_message(text):
    tokens = _load_tokens()
    if time.time() - tokens["obtained_at"] > tokens.get("expires_in", 21599) - 60:
        tokens = _refresh(tokens)

    template = json.dumps({
        "object_type": "text",
        "text": text,
        "link": {},
    })
    body = urllib.parse.urlencode({"template_object": template}).encode()
    req = urllib.request.Request(
        "https://kapi.kakao.com/v2/api/talk/memo/default/send", data=body, method="POST",
    )
    req.add_header("Authorization", f"Bearer {tokens['access_token']}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # one retry after a forced refresh, in case the token was invalid rather than just expired
        tokens = _refresh(tokens)
        req.headers["Authorization"] = f"Bearer {tokens['access_token']}"
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    auth_p = sub.add_parser("auth")
    auth_p.add_argument("--rest-api-key", required=True)
    auth_p.add_argument("--redirect-uri", required=True)
    auth_p.add_argument("--code", required=True)
    auth_p.add_argument("--client-secret", default=None)

    send_p = sub.add_parser("send")
    send_p.add_argument("text")

    args = parser.parse_args()
    if args.cmd == "auth":
        exchange_code(args.rest_api_key, args.redirect_uri, args.code, args.client_secret)
    elif args.cmd == "send":
        result = send_message(args.text)
        print(result)


if __name__ == "__main__":
    main()
