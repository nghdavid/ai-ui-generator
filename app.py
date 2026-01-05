import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

from flask import Flask, Response, flash, redirect, render_template, request, url_for
from openai import OpenAI
from dotenv import load_dotenv

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PROMPT_CHARS = 800

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev")


def _image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_prompt() -> str:
    return (
        "You are an expert frontend engineer. Recreate the UI from the screenshot as a single, "
        "self-contained HTML document. Use semantic HTML and a <style> tag for CSS. "
        "Do not use external assets, libraries, or links. If images are present, "
        "replace them with simple colored blocks or gradients. "
        "Match layout, spacing, colors, and typography as closely as possible. "
        "Return only the HTML document, with no markdown or commentary."
    )


def _build_prompt_from_text(user_prompt: str, existing_html: str = "") -> str:
    base_instructions = (
        "You are an expert frontend engineer. Use semantic HTML and a <style> tag for CSS. "
        "Do not use external assets, libraries, or links. If images are referenced, "
        "replace them with simple colored blocks or gradients. "
        "Return only the HTML document, with no markdown or commentary. "
    )

    if existing_html:
        return (
            f"{base_instructions}"
            "Update the existing HTML to reflect the user's requested changes. "
            "Preserve the overall structure where possible and return a full HTML document. "
            f"User request: {user_prompt}\n"
            "Existing HTML:\n<<<HTML\n"
            f"{existing_html}\n"
            "HTML"
        )

    return (
        f"{base_instructions}"
        "Create a single-screen UI based on the user prompt. "
        "Match layout, spacing, colors, and typography as closely as possible. "
        f"User prompt: {user_prompt}"
    )


def _build_vercel_readme() -> str:
    return (
        "Vercel deployment\n"
        "1) Unzip this folder.\n"
        "2) Run: npx vercel\n"
        "3) For production: npx vercel --prod\n"
    )


def _sanitize_project_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", name.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        cleaned = "html-export"
    if len(cleaned) > 48:
        cleaned = cleaned[:48].strip("-")
    return cleaned or "html-export"


def _default_project_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"html-export-{stamp}"


def _build_project_name(name: str) -> str:
    return _sanitize_project_name(name) if name else _default_project_name()


def _parse_vercel_error(body: str, fallback: str) -> str:
    try:
        payload = json.loads(body)
        if isinstance(payload, dict):
            if "error" in payload:
                error = payload["error"]
                if isinstance(error, dict):
                    return error.get("message") or error.get("code") or fallback
                if isinstance(error, str):
                    return error
            if "message" in payload and isinstance(payload["message"], str):
                return payload["message"]
    except json.JSONDecodeError:
        pass
    return fallback


def _deploy_to_vercel(html: str, token: str, project_name: str) -> tuple[str, str]:
    payload = {
        "name": project_name,
        "files": [
            {
                "file": "index.html",
                "data": html,
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.vercel.com/v13/deployments?skipAutoDetectionConfirmation=1",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "ignore")
        error_message = _parse_vercel_error(
            error_body, f"Vercel API error ({exc.code})."
        )
        return "", error_message
    except urllib.error.URLError as exc:
        return "", f"Network error contacting Vercel: {exc.reason}"
    except Exception as exc:
        return "", f"Unexpected error contacting Vercel: {exc}"

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        return "", "Vercel API returned an unexpected response."

    url = ""
    if isinstance(payload, dict):
        url = payload.get("url") or ""
        if not url and isinstance(payload.get("alias"), list):
            url = payload["alias"][0] if payload["alias"] else ""

    if not url:
        return "", "Vercel API did not return a deployment URL."
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url, ""


def _generate_html(image_bytes: bytes, mime_type: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    image_url = _image_to_data_url(image_bytes, mime_type)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("No HTML returned by the model")
    return content.strip()


def _generate_ui_html(prompt: str, existing_html: str = "") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": _build_prompt_from_text(prompt, existing_html=existing_html),
            }
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("No HTML returned by the model")
    return content.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prompt")
def prompt_ui():
    return render_template("prompt.html")


@app.route("/generate", methods=["POST"])
def generate():
    if "screenshot" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("index"))

    file = request.files["screenshot"]
    if not file or file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))

    if file.mimetype not in ALLOWED_MIME_TYPES:
        flash("Unsupported file type. Use PNG, JPG, or WEBP.")
        return redirect(url_for("index"))

    image_bytes = file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        flash("File too large. Limit is 10MB.")
        return redirect(url_for("index"))

    try:
        html = _generate_html(image_bytes, file.mimetype)
    except Exception as exc:
        flash(f"Error generating HTML: {exc}")
        return redirect(url_for("index"))

    return render_template("result.html", html=html)


@app.route("/modify-html", methods=["POST"])
def modify_html():
    prompt = request.form.get("prompt", "").strip()
    html = request.form.get("html", "").strip()

    if not html:
        flash("No HTML available to modify.")
        return redirect(url_for("index"))

    if not prompt:
        flash("Please enter a prompt.")
        return render_template("result.html", html=html, prompt=prompt)

    if len(prompt) > MAX_PROMPT_CHARS:
        flash("Prompt is too long. Please keep it under 800 characters.")
        return render_template("result.html", html=html, prompt=prompt)

    try:
        updated_html = _generate_ui_html(prompt, existing_html=html)
    except Exception as exc:
        flash(f"Error updating HTML: {exc}")
        return render_template("result.html", html=html, prompt=prompt)

    return render_template("result.html", html=updated_html, prompt=prompt)


@app.route("/generate-ui", methods=["POST"])
def generate_ui():
    prompt = request.form.get("prompt", "").strip()
    existing_html = request.form.get("html", "").strip()
    if not prompt:
        flash("Please enter a prompt.")
        return redirect(url_for("prompt_ui"))

    if len(prompt) > MAX_PROMPT_CHARS:
        flash("Prompt is too long. Please keep it under 800 characters.")
        return redirect(url_for("prompt_ui"))

    try:
        html = _generate_ui_html(prompt, existing_html=existing_html)
    except Exception as exc:
        flash(f"Error generating UI HTML: {exc}")
        return redirect(url_for("prompt_ui"))

    return render_template("prompt_result.html", html=html, prompt=prompt)


@app.route("/download", methods=["POST"])
def download():
    html = request.form.get("html", "")
    if not html:
        flash("No HTML available for download.")
        return redirect(url_for("index"))

    response = Response(html, mimetype="text/html")
    response.headers["Content-Disposition"] = "attachment; filename=generated.html"
    return response


@app.route("/download-vercel", methods=["POST"])
def download_vercel():
    html = request.form.get("html", "").strip()
    if not html:
        flash("No HTML available for Vercel export.")
        return redirect(url_for("index"))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("index.html", html)
        zip_file.writestr("README.txt", _build_vercel_readme())
        zip_file.writestr(".env_example", "VERCEL_TOKEN=your_token_here\n")

    zip_buffer.seek(0)
    response = Response(zip_buffer.read(), mimetype="application/zip")
    response.headers["Content-Disposition"] = "attachment; filename=vercel-deploy.zip"
    return response


def _render_deploy_result(
    source: str,
    html: str,
    prompt: str = "",
    deployment_url: str = "",
    deployment_error: str = "",
):
    template = "prompt_result.html" if source == "prompt" else "result.html"
    return render_template(
        template,
        html=html,
        prompt=prompt,
        deployment_url=deployment_url,
        deployment_error=deployment_error,
    )


@app.route("/deploy-vercel", methods=["POST"])
def deploy_vercel():
    html = request.form.get("html", "").strip()
    source = request.form.get("source", "screenshot").strip()
    prompt = request.form.get("prompt", "").strip()

    if not html:
        return _render_deploy_result(
            source,
            html,
            prompt,
            deployment_error="No HTML available for Vercel deploy.",
        )

    token = request.form.get("vercel_token", "").strip()
    if not token:
        token = os.getenv("VERCEL_TOKEN", "").strip()
    if not token:
        return _render_deploy_result(
            source,
            html,
            prompt,
            deployment_error="Provide a Vercel API token or set VERCEL_TOKEN.",
        )

    project_name = _build_project_name(request.form.get("project_name", "").strip())
    deployment_url, error_message = _deploy_to_vercel(html, token, project_name)
    if error_message:
        return _render_deploy_result(
            source,
            html,
            prompt,
            deployment_error=error_message,
        )

    return _render_deploy_result(
        source,
        html,
        prompt,
        deployment_url=deployment_url,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
