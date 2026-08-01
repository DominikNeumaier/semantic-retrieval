// @ts-check
import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

// Build a fully static site. base="/" so local dev works; for GitHub Enterprise
// Pages deployment under e.g. /MasterThesis/web/, set base via env or edit here.
export default defineConfig({
  integrations: [tailwind({ applyBaseStyles: true })],
  site: process.env.SITE_URL || "http://localhost:4321",
  base: process.env.BASE_PATH || "/",
  trailingSlash: "always",
  output: "static",
});
