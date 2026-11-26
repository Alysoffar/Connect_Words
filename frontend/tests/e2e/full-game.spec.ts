import { expect, test } from "@playwright/test";

test("plays through a complete connection and restores it after refresh", async ({
  page,
}) => {
  await page.route("**/api/puzzle/daily**", async (route) => {
    await route.fulfill({
      json: {
        puzzleId: "en-daily:2026-03-08",
        lang: "en",
        anchorA: "river",
        anchorB: "bank",
        threshold: 0.72,
      },
    });
  });

  await page.route("**/api/guess", async (route) => {
    const { word } = route.request().postDataJSON() as { word: string };
    const common = {
      puzzleId: "en-daily:2026-03-08",
      lang: "en",
      normalized: word,
      prunedEdges: [],
    };

    if (word === "water") {
      await route.fulfill({
        json: {
          ...common,
          word,
          edges: [
            {
              wordA: "water",
              wordB: "river",
              weight: 0.84,
              nearMiss: false,
              retained: true,
            },
          ],
          connected: false,
          path: null,
          score: null,
        },
      });
      return;
    }

    await route.fulfill({
      json: {
        ...common,
        word,
        edges: [
          {
            wordA: "shore",
            wordB: "water",
            weight: 0.81,
            nearMiss: false,
            retained: true,
          },
          {
            wordA: "shore",
            wordB: "bank",
            weight: 0.79,
            nearMiss: false,
            retained: true,
          },
        ],
        connected: true,
        path: ["river", "water", "shore", "bank"],
        score: 92,
      },
    });
  });

  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Inspect node river" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Inspect node bank" }),
  ).toBeVisible();

  await page.getByLabel("Enter a word").fill("water");
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(
    page.getByRole("button", { name: "Inspect node water" }),
  ).toBeVisible();
  await expect(page.getByText("3 / 50")).toBeVisible();

  await page.getByLabel("Enter a word").fill("shore");
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(
    page.getByRole("heading", { name: "Your connection" }),
  ).toBeVisible();
  await expect(
    page.getByText("Puzzle solved in 3 steps and 2 guesses."),
  ).toBeAttached();
  await expect(page.getByRole("progressbar")).toHaveAttribute(
    "aria-valuenow",
    "4",
  );

  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Your connection" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Inspect node shore" }),
  ).toBeVisible();
});
