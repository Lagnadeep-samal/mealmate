#!/usr/bin/env python3
"""
mealmate - a weekly meal planner that runs on a local open-weight model and
will not hand over a meal that contains something your housemate can't eat.

    python mealmate.py plan  --profile profile.json --days 5
    python mealmate.py plan  --profile profile.json --mock        # no model needed
    python mealmate.py check --profile profile.json "soy sauce, tahini, oat milk"
    python mealmate.py serve                                      # web app at http://localhost:8000

Standard library only. Talks to Ollama (https://ollama.com) on localhost.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from allergens import AllergenGuard, Hit, normalise

MODEL = os.environ.get("MEALMATE_MODEL", "qwen2.5:7b")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SLOTS = ["breakfast", "lunch", "dinner"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MAX_TRIES = 4

DAY_SCHEMA = {
    "type": "object",
    "properties": {
        "meals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "string", "enum": SLOTS},
                    "name": {"type": "string"},
                    "minutes": {"type": "integer"},
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item": {"type": "string"},
                                "amount": {"type": "number"},
                                "unit": {"type": "string"},
                            },
                            "required": ["item", "amount", "unit"],
                        },
                    },
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["slot", "name", "minutes", "ingredients", "steps"],
            },
        }
    },
    "required": ["meals"],
}


# ----------------------------------------------------------------------------
# Model clients. Anything with .chat(messages, schema) -> str works, so swapping
# models (or runtimes: llama.cpp server, LM Studio, vLLM) is a few lines.
# ----------------------------------------------------------------------------
class OllamaClient:
    def __init__(self, model: str = MODEL, host: str = HOST):
        self.model, self.host = model, host.rstrip("/")

    def chat(self, messages: list[dict], schema: dict) -> str:
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": False,
            "format": schema, "options": {"temperature": 0.6},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.load(resp)["message"]["content"]
        except urllib.error.URLError as e:
            sys.exit(f"Can't reach Ollama at {self.host} ({e.reason}).\n"
                     f"Start it with `ollama serve`, then `ollama pull {self.model}`.")


_SAFE_DAYS = [
    {"meals": [
        {"slot": "breakfast", "name": "Banana rice porridge", "minutes": 15,
         "ingredients": [{"item": "short grain rice", "amount": 1, "unit": "cup"},
                         {"item": "banana", "amount": 2, "unit": ""},
                         {"item": "cinnamon", "amount": 1, "unit": "tsp"},
                         {"item": "maple syrup", "amount": 2, "unit": "tbsp"}],
         "steps": ["Simmer the rice in 3 cups water for 20 minutes.", "Stir in mashed banana, cinnamon and syrup."]},
        {"slot": "lunch", "name": "Chickpea and tomato stew", "minutes": 30,
         "ingredients": [{"item": "chickpeas, cooked", "amount": 2, "unit": "cups"},
                         {"item": "tomatoes, chopped", "amount": 4, "unit": ""},
                         {"item": "onion", "amount": 1, "unit": ""},
                         {"item": "garlic", "amount": 3, "unit": "cloves"},
                         {"item": "ground cumin", "amount": 1, "unit": "tsp"},
                         {"item": "olive oil", "amount": 2, "unit": "tbsp"},
                         {"item": "baby spinach", "amount": 2, "unit": "handfuls"}],
         "steps": ["Soften onion and garlic in oil.", "Add cumin, tomatoes and chickpeas; simmer 20 minutes.", "Wilt in the spinach."]},
        {"slot": "dinner", "name": "Lemon herb chicken with roast potatoes", "minutes": 40,
         "ingredients": [{"item": "chicken thighs", "amount": 600, "unit": "g"},
                         {"item": "potatoes", "amount": 600, "unit": "g"},
                         {"item": "lemon", "amount": 1, "unit": ""},
                         {"item": "olive oil", "amount": 3, "unit": "tbsp"},
                         {"item": "fresh rosemary", "amount": 2, "unit": "sprigs"},
                         {"item": "green beans", "amount": 250, "unit": "g"}],
         "steps": ["Toss everything but the beans in oil and lemon.", "Roast at 220C for 35 minutes.", "Steam the beans and serve alongside."]},
    ]},
    {"meals": [
        {"slot": "breakfast", "name": "Potato and pepper hash", "minutes": 20,
         "ingredients": [{"item": "potatoes", "amount": 400, "unit": "g"},
                         {"item": "red bell pepper", "amount": 1, "unit": ""},
                         {"item": "onion", "amount": 1, "unit": ""},
                         {"item": "olive oil", "amount": 2, "unit": "tbsp"},
                         {"item": "smoked paprika", "amount": 1, "unit": "tsp"}],
         "steps": ["Dice and par-boil the potatoes.", "Fry with onion and pepper until crisp, then add paprika."]},
        {"slot": "lunch", "name": "Red lentil soup", "minutes": 30,
         "ingredients": [{"item": "red lentils", "amount": 1.5, "unit": "cups"},
                         {"item": "carrots", "amount": 2, "unit": ""},
                         {"item": "onion", "amount": 1, "unit": ""},
                         {"item": "ground cumin", "amount": 1, "unit": "tsp"},
                         {"item": "vegetable stock", "amount": 1, "unit": "litre"}],
         "steps": ["Sweat onion and carrot.", "Add lentils, cumin and stock; simmer 20 minutes and blend."]},
        {"slot": "dinner", "name": "Beef and broccoli rice bowls", "minutes": 25,
         "ingredients": [{"item": "beef strips", "amount": 400, "unit": "g"},
                         {"item": "broccoli", "amount": 1, "unit": "head"},
                         {"item": "short grain rice", "amount": 1.5, "unit": "cups"},
                         {"item": "fresh ginger", "amount": 1, "unit": "thumb"},
                         {"item": "garlic", "amount": 2, "unit": "cloves"},
                         {"item": "lime", "amount": 1, "unit": ""}],
         "steps": ["Cook the rice.", "Stir-fry beef with ginger and garlic, add broccoli, finish with lime."]},
    ]},
    {"meals": [
        {"slot": "breakfast", "name": "Sweet potato and avocado plate", "minutes": 20,
         "ingredients": [{"item": "sweet potato", "amount": 2, "unit": ""},
                         {"item": "avocado", "amount": 1, "unit": ""},
                         {"item": "lime", "amount": 1, "unit": ""},
                         {"item": "olive oil", "amount": 1, "unit": "tbsp"}],
         "steps": ["Slice and pan-fry the sweet potato.", "Top with avocado and a squeeze of lime."]},
        {"slot": "lunch", "name": "Chicken and rice salad", "minutes": 20,
         "ingredients": [{"item": "cooked chicken", "amount": 300, "unit": "g"},
                         {"item": "cooked rice", "amount": 2, "unit": "cups"},
                         {"item": "cucumber", "amount": 1, "unit": ""},
                         {"item": "cherry tomatoes", "amount": 200, "unit": "g"},
                         {"item": "lemon juice", "amount": 2, "unit": "tbsp"},
                         {"item": "olive oil", "amount": 2, "unit": "tbsp"}],
         "steps": ["Chop everything and toss with lemon juice and oil."]},
        {"slot": "dinner", "name": "Turkey meatballs with zucchini", "minutes": 35,
         "ingredients": [{"item": "ground turkey", "amount": 500, "unit": "g"},
                         {"item": "onion", "amount": 1, "unit": ""},
                         {"item": "fresh parsley", "amount": 1, "unit": "bunch"},
                         {"item": "zucchini", "amount": 3, "unit": ""},
                         {"item": "tomatoes, chopped", "amount": 4, "unit": ""},
                         {"item": "olive oil", "amount": 2, "unit": "tbsp"}],
         "steps": ["Mix turkey, grated onion and parsley; roll into balls.", "Brown, then simmer in tomatoes for 15 minutes.", "Serve over sauteed zucchini."]},
    ]},
]

_UNSAFE_DAY = {"meals": [
    {"slot": "breakfast", "name": "Buttered toast with scrambled eggs", "minutes": 10,
     "ingredients": [{"item": "bread", "amount": 4, "unit": "slices"},
                     {"item": "butter", "amount": 2, "unit": "tbsp"},
                     {"item": "eggs", "amount": 4, "unit": ""}], "steps": ["Toast, butter, scramble."]},
    {"slot": "lunch", "name": "Hummus veggie wrap", "minutes": 10,
     "ingredients": [{"item": "chickpeas", "amount": 1, "unit": "cup"},
                     {"item": "cucumber", "amount": 1, "unit": ""}],
     "steps": ["Blend chickpeas, then drizzle generously with tahini and roll up."]},
    {"slot": "dinner", "name": "Pad thai", "minutes": 25,
     "ingredients": [{"item": "rice noodles", "amount": 250, "unit": "g"},
                     {"item": "shrimp", "amount": 300, "unit": "g"},
                     {"item": "crushed peanuts", "amount": 0.5, "unit": "cup"},
                     {"item": "fish sauce", "amount": 2, "unit": "tbsp"}], "steps": ["Stir-fry and serve."]},
]}


class MockClient:
    """Offline stand-in. Answers each day badly first, then well, so you can watch
    the allergen guard reject and the retry loop recover. Needs no model."""

    def __init__(self):
        self.day = -1

    def chat(self, messages: list[dict], schema: dict) -> str:
        if messages[-1]["content"].startswith("Rejected"):
            return json.dumps(_SAFE_DAYS[self.day % len(_SAFE_DAYS)])
        self.day += 1
        return json.dumps(_UNSAFE_DAY)


# ----------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------
def system_prompt(profile: dict, guard: AllergenGuard) -> str:
    who = profile.get("name", "a housemate")
    lines = [
        f"You plan home-cooked meals for a household of {profile.get('household_size', 2)}.",
        f"{who} has food allergies. These are medical, not preferences. Never include "
        f"anything that contains or is derived from: {guard.describe()}.",
        "Watch for hidden sources: for example milk hides in butter, ghee, whey and casein; "
        "soy sauce contains soy and wheat; fish sauce and Worcestershire sauce contain fish; "
        "pesto contains nuts and cheese; hummus contains sesame.",
        "If the classic version of a dish needs a forbidden ingredient, choose a different dish.",
        "Use simple whole ingredients. Name every ingredient specifically; never write "
        "'sauce', 'spice mix' or 'seasoning' on its own.",
        f"Each meal must take {profile.get('max_cook_minutes', 40)} minutes or less.",
        "Ingredient amounts are for the whole household. Reply with JSON only.",
    ]
    if profile.get("diet"):
        lines.append(f"Diet: {profile['diet']}.")
    if profile.get("likes"):
        lines.append(f"Favourite foods (use often): {', '.join(profile['likes'])}.")
    if profile.get("dislikes"):
        lines.append(f"Dislikes (never use): {', '.join(profile['dislikes'])}.")
    if profile.get("notes"):
        lines.append(profile["notes"])
    return "\n".join(lines)


def validate_day(data, guard: AllergenGuard) -> tuple[list[str], list[Hit]]:
    """Returns (problems that force a retry, cautions to show the person)."""
    problems: list[str] = []
    cautions: list[Hit] = []
    meals = data.get("meals") if isinstance(data, dict) else None
    if not isinstance(meals, list):
        return ["The reply had no 'meals' list."], cautions

    got = [str(m.get("slot")) for m in meals if isinstance(m, dict)]
    if sorted(got) != sorted(SLOTS):
        problems.append(f"Expected exactly one each of {', '.join(SLOTS)}; got {got}.")

    for m in meals:
        if not isinstance(m, dict):
            continue
        texts = [m.get("name", "")] + [str(s) for s in m.get("steps", [])]
        texts += [str(i.get("item", "")) if isinstance(i, dict) else str(i)
                  for i in m.get("ingredients", [])]
        for t in texts:
            for h in guard.scan(t):
                (problems if h.level == "block" else cautions).append(
                    f"{m.get('name', '?')}: {h}" if h.level == "block" else h)
    return problems, cautions


def plan_day(client, guard, profile, day_name, used_names, log=print):
    msgs = [
        {"role": "system", "content": system_prompt(profile, guard)},
        {"role": "user", "content": (
            f"Plan {day_name}: breakfast, lunch and dinner. "
            + (f"Already planned this week, so don't repeat: {', '.join(used_names)}. " if used_names else "")
            + (f"Pantry staples already at home: {', '.join(profile['pantry'])}. " if profile.get("pantry") else "")
        )},
    ]
    problems: list[str] = []
    for attempt in range(1, MAX_TRIES + 1):
        raw = client.chat(msgs, DAY_SCHEMA)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data, problems = None, ["The reply was not valid JSON."]
        else:
            problems, cautions = validate_day(data, guard)
            if not problems:
                if attempt > 1:
                    log(f"    ok on attempt {attempt}")
                return data, cautions, []
        log(f"    attempt {attempt} rejected: {problems[0]}" + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""))
        msgs += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                "Rejected by the allergy checker:\n- " + "\n- ".join(problems[:8])
                + "\nRewrite the whole day without any of these. Do not just rename them; "
                  "choose different dishes if needed. Check the cooking steps too, not only the ingredient list.")},
        ]
    return None, [], problems


def build_plan(client, profile: dict, days: int, log=print) -> dict:
    guard = AllergenGuard(profile.get("allergies", []), profile.get("also_avoid", []),
                          strict=profile.get("treat_cautions_as_blocking", False))
    if not guard.categories and not guard.custom:
        sys.exit("The profile lists no allergies. Add some to 'allergies' in profile.json.")
    for u in guard.unrecognised:
        log(f"  note: '{u}' isn't a built-in category, so it's treated as a literal word to avoid.")

    pantry = {normalise(p) for p in profile.get("pantry", [])}
    plan = {"person": profile.get("name", "your housemate"), "checked_for": guard.describe(),
            "days": [], "failed": [], "cautions": [], "label_checks": [], "grocery": []}
    used: list[str] = []
    totals: dict[tuple[str, str], float] = defaultdict(float)
    labels: set[str] = set()
    cautions: set[str] = set()

    for i in range(days):
        day = WEEKDAYS[i % 7]
        log(f"  {day}...")
        data, cs, problems = plan_day(client, guard, profile, day, used, log)
        if data is None:
            plan["failed"].append({"day": day, "reasons": problems[:3]})
            log(f"    gave up on {day}; leaving it blank rather than risk it.")
            continue
        plan["days"].append({"day": day, "meals": sorted(data["meals"], key=lambda m: SLOTS.index(m["slot"]))})
        for c in cs:
            cautions.add(f"{c.term} (in \"{c.text}\") may relate to {c.allergen.replace('_', ' ')}")
        for m in data["meals"]:
            used.append(m["name"])
            for ing in m["ingredients"]:
                name = normalise(ing["item"])
                if guard.label_flags(ing["item"]):
                    labels.add(ing["item"].strip().lower())
                if name in pantry:
                    continue
                try:
                    amt = float(ing["amount"])
                except (TypeError, ValueError):
                    amt = 1.0
                totals[(ing["item"].strip().lower(), str(ing["unit"]).strip().lower())] += amt

    plan["cautions"] = sorted(cautions)
    plan["label_checks"] = sorted(labels)
    plan["grocery"] = [{"item": k[0], "amount": round(v, 2), "unit": k[1]} for k, v in sorted(totals.items())]
    return plan


# ----------------------------------------------------------------------------
# A page you can stick on the fridge
# ----------------------------------------------------------------------------
_CSS = """
:root{--paper:#f8fafc;--ink:#0f172a;--muted:#64748b;--rule:#e2e8f0;--card:#ffffff;--accent:#4f46e5;--accent-2:#7c3aed;--accent-soft:#eef2ff;--warn:#b45309;--warn-soft:#fffbeb;--bad:#e11d48}
@media (prefers-color-scheme:dark){:root{--paper:#0b1020;--ink:#e2e8f0;--muted:#94a3b8;--rule:#1e293b;--card:#111a2e;--accent:#818cf8;--accent-2:#a78bfa;--accent-soft:#1a2347;--warn:#fbbf24;--warn-soft:#2a2110;--bad:#fb7185}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,sans-serif}
main{max-width:1100px;margin:0 auto;padding:clamp(16px,4vw,48px)}
h1,h2,h3{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,sans-serif;letter-spacing:-.02em;line-height:1.15;margin:0}
h1{font-size:clamp(2rem,4.5vw,3rem);font-weight:800;background:linear-gradient(90deg,var(--accent),var(--accent-2));-webkit-background-clip:text;background-clip:text;color:transparent;width:fit-content}
h2{font-size:1.4rem;font-weight:700;margin:2.6rem 0 .8rem}
h3{font-size:1.1rem;font-weight:650}
.guard{margin:.7rem 0 0;color:var(--muted);max-width:62ch}
.guard b{color:var(--accent);background:var(--accent-soft);padding:.05em .45em;border-radius:6px;font-weight:600}
.grid{display:grid;gap:1px;overflow-x:auto;background:var(--rule);border:1px solid var(--rule);border-radius:14px;margin-top:1.4rem;overflow:hidden}
.cell{background:var(--card);padding:.8rem .95rem}
.cell .slot{display:block;color:var(--muted);font-size:.82rem}
.cell .dish{font-weight:600}
.cell.head{background:var(--accent);color:#fff;font-weight:600}
.recipes{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.2rem}
article{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:1.1rem 1.2rem;break-inside:avoid;box-shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px -12px rgba(79,70,229,.18)}
article .when{display:inline-block;color:var(--accent);background:var(--accent-soft);font-size:.8rem;font-weight:600;padding:.1rem .6rem;border-radius:999px;margin-bottom:.5rem}
article ul,article ol{margin:.4rem 0 .6rem;padding-left:1.2rem}
article li::marker{color:var(--accent)}
.cols{columns:2 260px;column-gap:2rem;padding-left:1.1rem}
.notice{border-radius:12px;border:1px solid color-mix(in srgb,var(--warn) 35%,transparent);background:var(--warn-soft);padding:.7rem 1.1rem;margin:1rem 0}
.fail{border-color:color-mix(in srgb,var(--bad) 40%,transparent)}
footer{margin-top:3rem;color:var(--muted);font-size:.9rem;max-width:70ch}
@media print{body{background:#fff;color:#000}h1{background:none;color:#000}.cell.head{background:#222}article{box-shadow:none;border-color:#999}.grid{border-color:#999}}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _num(x) -> str:
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return ""


def render_html(plan: dict) -> str:
    who = _e(plan["person"])
    days = plan["days"]
    out = [f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Meals for the week</title><style>{_CSS}</style><main>",
           "<h1>This week's meals</h1>",
           f"<p class=guard>Every dish was checked against <b>{_e(plan['checked_for'])}</b> for {who} "
           f"before it got on this page. Label reading is still on you.</p>"]

    if plan["failed"]:
        for f in plan["failed"]:
            out.append(f"<div class='notice fail'><b>{_e(f['day'])} is blank.</b> The model kept suggesting something unsafe "
                       f"({_e(f['reasons'][0])}). Plan that day by hand, or try a larger model.</div>")

    out.append(f"<div class=grid style='grid-template-columns:repeat({max(len(days), 1)},minmax(0,1fr))'>")
    for d in days:
        out.append(f"<div class='cell head'>{_e(d['day'])}</div>")
    for slot in SLOTS:
        for d in days:
            m = next((m for m in d["meals"] if m["slot"] == slot), None)
            out.append(f"<div class=cell><span class=slot>{slot}</span><span class=dish>{_e(m['name']) if m else '-'}</span></div>")
    out.append("</div>")

    out.append("<h2>Recipes</h2><div class=recipes>")
    for d in days:
        for m in d["meals"]:
            ings = "".join(f"<li>{_e(' '.join(str(x) for x in (_num(i['amount']), i['unit'], i['item']) if str(x)))}</li>"
                           for i in m["ingredients"])
            steps = "".join(f"<li>{_e(s)}</li>" for s in m["steps"])
            out.append(f"<article><div class=when>{_e(d['day'])} {m['slot']}, {_e(m['minutes'])} min</div>"
                       f"<h3>{_e(m['name'])}</h3><ul>{ings}</ul><ol>{steps}</ol></article>")
    out.append("</div>")

    out.append("<h2>Shopping list</h2><ul class=cols>")
    for g in plan["grocery"]:
        amt = f"{g['amount']:g}"
        out.append(f"<li>{_e(g['item'])} <span style='color:var(--muted)'>{amt} {_e(g['unit'])}</span></li>")
    out.append("</ul>")

    if plan["label_checks"]:
        out.append("<h2>Read the label on these</h2>"
                   f"<div class=notice>Packaged versions of these can hide an allergen. Check the ingredients and any "
                   f"&ldquo;may contain&rdquo; line before buying.<ul class=cols>"
                   + "".join(f"<li>{_e(x)}</li>" for x in plan["label_checks"]) + "</ul></div>")
    if plan["cautions"]:
        out.append(f"<h2>Ask {who}</h2><div class=notice>These aren't blocked, but some people with the same allergy avoid them:<ul>"
                   + "".join(f"<li>{_e(x)}</li>" for x in plan["cautions"]) + "</ul></div>")

    out.append("<footer>Made by a local open-weight model, then checked by a plain word list. "
               "The checker can't see cross-contact or what's inside a packet, and it isn't medical advice. "
               "For severe allergies, have the person with the allergy read each plan before cooking.</footer></main></html>")
    return "".join(out)



# ----------------------------------------------------------------------------
# Local web app:  python mealmate.py serve   ->  http://localhost:8000
# Binds to 127.0.0.1 only, so nothing is reachable from outside this machine.
# ----------------------------------------------------------------------------
APP_HTML = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>mealmate</title>
<style>
:root{--paper:#f8fafc;--ink:#0f172a;--muted:#64748b;--rule:#e2e8f0;--card:#fff;--accent:#4f46e5;--accent-2:#7c3aed;--soft:#eef2ff}
@media (prefers-color-scheme:dark){:root{--paper:#0b1020;--ink:#e2e8f0;--muted:#94a3b8;--rule:#1e293b;--card:#111a2e;--accent:#818cf8;--accent-2:#a78bfa;--soft:#1a2347}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:1.5rem;max-width:1400px;margin:0 auto;padding:clamp(16px,3vw,36px)}
@media (max-width:860px){.wrap{grid-template-columns:1fr}}
h1{margin:0 0 .2rem;font-size:2rem;font-weight:800;letter-spacing:-.02em;background:linear-gradient(90deg,var(--accent),var(--accent-2));-webkit-background-clip:text;background-clip:text;color:transparent;width:fit-content}
p.sub{margin:0 0 1.2rem;color:var(--muted)}
form{background:var(--card);border:1px solid var(--rule);border-radius:14px;padding:1.1rem;display:grid;gap:.8rem;align-content:start}
label{display:grid;gap:.25rem;font-weight:600;font-size:.9rem}
label small{font-weight:400;color:var(--muted)}
input{font:inherit;padding:.55rem .7rem;border:1px solid var(--rule);border-radius:10px;background:var(--paper);color:var(--ink)}
input:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
button{font:inherit;font-weight:700;color:#fff;border:0;border-radius:10px;padding:.75rem 1rem;cursor:pointer;background:linear-gradient(90deg,var(--accent),var(--accent-2))}
button:disabled{opacity:.55;cursor:progress}
#log{margin:0;background:var(--soft);color:var(--ink);border-radius:10px;padding:.7rem .9rem;font:13px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;max-height:200px;overflow:auto;min-height:3.2rem}
.err{color:#e11d48;font-weight:600}
iframe{width:100%;height:calc(100vh - 72px);min-height:560px;border:1px solid var(--rule);border-radius:14px;background:var(--card)}
.empty{display:grid;place-items:center;height:60vh;border:1px dashed var(--rule);border-radius:14px;color:var(--muted);text-align:center;padding:2rem}
</style>
<div class=wrap>
 <section>
  <h1>mealmate</h1>
  <p class=sub>Runs on your computer. Nothing is sent anywhere.</p>
  <form id=f>
   <label>Housemate's name<input id=name required></label>
   <label>Allergies<small>comma separated, e.g. peanut, dairy, shellfish</small><input id=allergies required></label>
   <label>Also avoid<small>not allergies, just no</small><input id=avoid></label>
   <label>Favourite foods<input id=likes></label>
   <label>Dislikes<input id=dislikes></label>
   <div class=row>
    <label>Days<input id=days type=number min=1 max=7 value=5></label>
    <label>People<input id=size type=number min=1 max=12 value=2></label>
   </div>
   <button id=go>Plan the week</button>
   <pre id=log aria-live=polite>Ready.</pre>
  </form>
 </section>
 <section id=out><div class=empty>Your plan will show up here.<br>The first one can take a few minutes on a laptop.</div></section>
</div>
<script>
const $=i=>document.getElementById(i), list=v=>v.split(",").map(x=>x.trim()).filter(Boolean);
let profile={};
fetch("/api/profile").then(r=>r.json()).then(p=>{
  profile=p; $("name").value=p.name||""; $("allergies").value=(p.allergies||[]).join(", ");
  $("avoid").value=(p.also_avoid||[]).join(", "); $("likes").value=(p.likes||[]).join(", ");
  $("dislikes").value=(p.dislikes||[]).join(", "); $("size").value=p.household_size||2;
});
$("f").addEventListener("submit",async e=>{
  e.preventDefault(); $("go").disabled=true; $("log").textContent="Starting...";
  const body={days:+$("days").value,profile:{...profile,name:$("name").value.trim(),allergies:list($("allergies").value),
    also_avoid:list($("avoid").value),likes:list($("likes").value),dislikes:list($("dislikes").value),household_size:+$("size").value}};
  await fetch("/api/plan",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const t=setInterval(async()=>{
    const s=await (await fetch("/api/status")).json();
    $("log").innerHTML=""; $("log").append(document.createTextNode(s.log.join("\\n")||"Working..."));
    if(s.error){const d=document.createElement("div");d.className="err";d.textContent=s.error;$("log").append(d)}
    if(!s.running){clearInterval(t);$("go").disabled=false;
      if(s.done)$("out").innerHTML='<iframe title="Meal plan" src="/plan?'+Date.now()+'"></iframe>'}
  },1000);
});
</script></html>"""


def serve(profile_path: str, port: int, model: str, mock: bool) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    try:
        base = json.loads(Path(profile_path).read_text())
    except FileNotFoundError:
        base = {"name": "", "allergies": [], "household_size": 2}
    state = {"running": False, "log": [], "html": "", "error": ""}
    lock = threading.Lock()

    def work(profile: dict, days: int) -> None:
        try:
            client = MockClient() if mock else OllamaClient(model)
            plan = build_plan(client, profile, days, log=lambda m: state["log"].append(m))
            state["html"] = render_html(plan)
        except SystemExit as e:  # clients and build_plan exit with a readable message
            state["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - show it in the page instead of dying
            state["error"] = f"{type(e).__name__}: {e}"
        finally:
            state["running"] = False

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: str, ctype: str) -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, APP_HTML, "text/html")
            elif path == "/api/profile":
                self._send(200, json.dumps(base), "application/json")
            elif path == "/api/status":
                self._send(200, json.dumps({"running": state["running"], "log": state["log"],
                                            "error": state["error"], "done": bool(state["html"])}),
                           "application/json")
            elif path == "/plan" and state["html"]:
                self._send(200, state["html"], "text/html")
            else:
                self._send(404, "Not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/plan":
                return self._send(404, "Not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                profile, days = dict(req["profile"]), max(1, min(7, int(req.get("days", 5))))
            except (ValueError, KeyError, TypeError):
                return self._send(400, "Bad request", "text/plain")
            with lock:
                if state["running"]:
                    return self._send(409, "Already planning", "text/plain")
                state.update(running=True, log=[], html="", error="")
            Path(profile_path).write_text(json.dumps(profile, indent=2))  # remember for next time
            threading.Thread(target=work, args=(profile, days), daemon=True).start()
            self._send(202, "started", "text/plain")

        def log_message(self, *args):  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mealmate is running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nStopped.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def load_profile(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        sys.exit(f"No profile at {path}. Copy profile.example.json to profile.json and edit it.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="generate a week of meals")
    p.add_argument("--profile", default="profile.json")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--out", default="plan", help="output name; writes <out>.html and <out>.json")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--mock", action="store_true", help="fake model that fails first, to demo the guard")

    w = sub.add_parser("serve", help="run the web app at http://localhost:8000")
    w.add_argument("--profile", default="profile.json")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--model", default=MODEL)
    w.add_argument("--mock", action="store_true")

    c = sub.add_parser("check", help="check ingredients or a recipe against the profile")
    c.add_argument("--profile", default="profile.json")
    c.add_argument("text", help="comma-separated ingredients or any recipe text")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return serve(a.profile, a.port, a.model, a.mock)
    profile = load_profile(a.profile)

    if a.cmd == "check":
        g = AllergenGuard(profile.get("allergies", []), profile.get("also_avoid", []),
                          strict=profile.get("treat_cautions_as_blocking", False))
        bad = False
        for part in [x for x in a.text.split(",") if x.strip()]:
            hits = g.scan(part.strip())
            flags = g.label_flags(part.strip())
            mark = "BLOCK  " if any(h.level == "block" for h in hits) else ("ask    " if hits else "ok     ")
            bad |= mark.startswith("BLOCK")
            extra = "; ".join(str(h) for h in hits) + (f"  [check label: {', '.join(flags)}]" if flags and not hits else "")
            print(f"{mark}{part.strip()}  {extra}".rstrip())
        sys.exit(1 if bad else 0)

    client = MockClient() if a.mock else OllamaClient(a.model)
    print(f"Planning {a.days} days with {'the mock model' if a.mock else a.model}...")
    plan = build_plan(client, profile, a.days)
    Path(f"{a.out}.json").write_text(json.dumps(plan, indent=2))
    Path(f"{a.out}.html").write_text(render_html(plan))
    print(f"Done: {a.out}.html ({len(plan['days'])} days, {len(plan['failed'])} blank). Open it in a browser or print it.")


if __name__ == "__main__":
    main()
