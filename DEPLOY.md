# Deploying Styla

Three pieces, all on free tiers:

| Piece | Runs on | Why |
| --- | --- | --- |
| Website | Netlify | Static files plus a small server function for page rendering. |
| API | Hugging Face Docker Space | Loads FashionCLIP, Segformer and rembg: about 3 GB of memory and a minute to start. The free CPU Space gives 2 vCPU and 16 GB. |
| Database | Neon | Already running; nothing to do. |

**The API cannot go on Netlify.** Netlify functions are capped near 250 MB and
time out in seconds, while the models alone are over a gigabyte.

---

## 1. The API, on a Hugging Face Space

1. Sign in at <https://huggingface.co> and choose **New → Space**.
2. Name it (for example `styla-api`), pick **Docker → Blank**, hardware
   **CPU basic (free)**, and make it **Public**.
3. Open the Space's **Files** tab and edit `README.md` so it begins with:

   ```yaml
   ---
   title: Styla API
   emoji: 👗
   colorFrom: pink
   colorTo: purple
   sdk: docker
   app_port: 7860
   ---
   ```

   `app_port: 7860` is what tells the Space which port to serve.

4. **Settings → Variables and secrets**, add these as **Secrets**:

   | Name | Value |
   | --- | --- |
   | `DATABASE_URL` | the Neon connection string from your local `.env` |
   | `STYLA_JWT_SECRET` | any long random string — set it, or every restart signs everyone out |
   | `HF_TOKEN` | your Hugging Face token (used by virtual try-on) |
   | `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | optional, improves shop results |

   And these as plain **Variables** (fill the Netlify address in step 3):

   | Name | Value |
   | --- | --- |
   | `PUBLIC_BASE_URL` | `https://<your-name>-styla-api.hf.space` |
   | `FRONTEND_URL` | your Netlify address |
   | `ALLOWED_ORIGINS` | your Netlify address |

5. Push this repository to the Space:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-name>/styla-api
   git push space main
   ```

   The first build takes 10–20 minutes: it installs CPU-only PyTorch and bakes
   the model weights into the image. Watch the **Logs** tab. When it prints
   `Application startup complete`, open
   `https://<your-name>-styla-api.hf.space/health` — it should answer
   `{"status":"ok"}`.

## 2. The website, on Netlify

1. Sign in at <https://netlify.com> → **Add new site → Import an existing
   project** → GitHub → `its-darya/Styla_stylist`.
2. Leave the build settings alone. `netlify.toml` already sets the base
   directory, the build command, the publish directory and the Nitro preset.
3. Before the first deploy, open **Site configuration → Environment variables**
   and add:

   | Name | Value |
   | --- | --- |
   | `VITE_API_URL` | `https://<your-name>-styla-api.hf.space` |

   Without it the site calls `localhost:8000` and every request fails.
4. Deploy. Netlify gives you an address like
   `https://styla-stylist.netlify.app`.

## 3. Introduce them to each other

Put the Netlify address into the Space's `FRONTEND_URL` and `ALLOWED_ORIGINS`
variables, then **Restart this Space**. Without this the browser blocks every
API call as a cross-origin request.

---

## What works, and what to expect

- **The seeded wardrobe travels with the API.** The garment photos are
  committed, so a fresh deploy already has the full wardrobe and every page
  works immediately.
- **Newly uploaded garments do not survive a rebuild.** A free Space has no
  permanent disk: when it rebuilds, uploads are gone and the committed
  wardrobe returns. Database rows survive (they are in Neon), so such items
  would show a missing picture. Fixes, when you want one:
  - Space **persistent storage**, about $5 a month, nothing to change in code; or
  - **Cloudflare R2** (free up to 10 GB) — I change uploads to write there.
- **The Space sleeps** after about 48 hours idle and wakes on the next visit,
  taking a minute or so to reload the models.
- **Virtual try-on** still depends on the shared GPU quota of the hosted
  try-on Spaces; when it runs out, Styla falls back to its own preview.

## Security

- `.env` is not committed and must never be. Every secret is pasted into the
  host's own settings.
- The Hugging Face token shared earlier in chat should be rotated at
  <https://huggingface.co/settings/tokens>.
- `STYLA_DEMO_EMAIL` / `STYLA_DEMO_PASSWORD` create the public demo account.
  Change the password before sharing the site, or the demo wardrobe is open to
  anyone who reads this repository.

## Running the container yourself

```bash
docker build -t styla-api .
docker run -p 7860:7860 --env-file .env styla-api
```
