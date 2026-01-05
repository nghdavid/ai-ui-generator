import base64
import os

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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
