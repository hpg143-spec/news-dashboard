# Signal: AI-filtered news by impact (free tier)

1. Create a public GitHub repo and push this folder.
2. Get a free key at https://aistudio.google.com/apikey, then add it as repo secret `GEMINI_API_KEY`.
3. Edit `feeds.json`: replace `YOUR+REGION` in the Google News URL, add more feeds (tier: primary / wire / general).
4. Actions tab -> "update-news" -> Run workflow. It then runs every 3 hours.
5. Settings -> Pages -> deploy from branch `main`, folder `/docs`.

Local test: `pip install -r requirements.txt && GEMINI_API_KEY=... python pipeline.py`, then `cd docs && python -m http.server`.
Set `GEMINI_MODEL` env var if the default model name changes or its free quota is too low.
