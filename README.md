# data_capstone

AI-powered UI generation app that turns screenshots or text prompts into self-contained HTML/CSS, then optionally packages or deploys the result to Vercel.

## Overview

This project uses the OpenAI API to generate frontend UI code in two ways:

- `Screenshot to HTML`: upload a UI screenshot and generate a matching single-file HTML page.
- `Prompt to HTML`: describe a UI in natural language and generate the interface from text.
- `Iterative AI refinement`: update generated HTML by submitting follow-up prompts that modify the existing output.

The generated result is designed to be portable:

- semantic HTML
- embedded CSS in a single file
- no external assets, libraries, or links required

## AI Features

- Multimodal generation from uploaded screenshots
- Natural-language UI generation from text prompts
- Prompt-driven editing of existing HTML outputs
- Constrained generation for clean, self-contained frontend code
- Fast export and optional deployment flow for generated pages

## Tech Stack

- Python
- Flask
- OpenAI API
- HTML/CSS
- Vercel Deployment API

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and add your credentials if needed:

```bash
cp .env_example .env
```

## Environment Variables

- `OPENAI_API_KEY`: required for AI HTML generation
- `OPENAI_MODEL`: optional, defaults to `gpt-4o-mini`
- `VERCEL_TOKEN`: optional, used for API deployment
- `FLASK_SECRET_KEY`: optional, used by Flask

## Run Locally

```bash
python app.py
```

Open [http://localhost:3000](http://localhost:3000)

## How It Works

### 1. Generate HTML from a Screenshot

- Upload a PNG, JPG, or WEBP file on the home page.
- The app sends the image to the OpenAI API.
- The model returns a self-contained HTML document that recreates the UI layout and styling.

### 2. Generate HTML from a Prompt

- Open the prompt workflow.
- Describe the interface you want in natural language.
- The app returns a full single-screen HTML/CSS UI.

### 3. Refine the Result with AI

- Submit follow-up prompts to update the generated HTML.
- The app preserves the existing structure where possible and regenerates the full document.

### 4. Export or Deploy

- Download the generated HTML directly
- Export a Vercel-ready zip bundle
- Deploy through the Vercel API with a personal access token

## Vercel Deployment Options

### Download ZIP

No token required.

1. Click `Deploy to Vercel`
2. Unzip the bundle
3. Run:

```bash
npx vercel
```

For production:

```bash
npx vercel --prod
```

### Deploy via API

- Paste a Vercel personal access token into the deploy form, or set `VERCEL_TOKEN`
- Optionally provide a project name
- Click `Deploy via API`
- The app returns a deployment URL

Create a token here: [https://vercel.com/account/tokens](https://vercel.com/account/tokens)

## Constraints

- Supported image types: PNG, JPG, WEBP
- Maximum image size: 10 MB
- Maximum prompt length: 800 characters

## Notes

- Tokens are used only during deployment and are not stored
- Generated HTML is self-contained for easy export and hosting
- Deployment is optional; the main focus of the project is AI-assisted UI generation
