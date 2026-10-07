# mealmate

A weekly meal planner for a housemate with food allergies. A local open-weight model
proposes meals; a plain word-list checker decides whether they are allowed. Nothing
leaves the laptop.

## Run it

```bash
# 1. a local model (one-time)
ollama pull qwen2.5:7b          # older laptop? try llama3.2:3b

# 2. describe your housemate
cp profile.example.json profile.json     # edit allergies, likes, dislikes

# 3a. use it in the browser (recommended)
python mealmate.py serve                 # then open http://localhost:8000

# 3b. or from the terminal
python mealmate.py plan --days 5         # writes plan.html and plan.json

# no model handy? watch the guard work with a fake one that gets it wrong first
python mealmate.py plan --profile profile.example.json --mock

# quick check of any recipe or shopping list
python mealmate.py check "soy sauce, pesto, oat milk"

# tests (no model or network needed)
python -m unittest -v
```

The web app listens on 127.0.0.1 only, so nothing outside your computer can reach it. It saves
the form back to `profile.json`. Generating takes a few minutes on a laptop without a GPU
(each day is a separate model call); `llama3.2:3b` is faster but plans less carefully.

Python 3.9+, standard library only. Use another model with `MEALMATE_MODEL=...` or
`--model ...`. Any runtime that can take a JSON schema works if you add a client with a
`chat(messages, schema)` method.

## How it stays safe

1. The model plans one day at a time, with the allergies in its system prompt.
2. `allergens.py` scans every dish name, ingredient **and cooking step**. It knows hidden
   sources (ghee, casein, tahini, Worcestershire, pesto) and avoids false alarms
   (peanut butter is not dairy, eggplant is not egg).
3. On a hit, the model is told exactly what was rejected and tries again, up to four times.
4. If it never gets there, that day is left blank. Unsafe output is never shown.
5. An allergy it doesn't recognise (say "kiwi") is treated as a literal word to avoid,
   never silently ignored.
6. Packaged things (stock, sauces, spice mixes) go on a "read the label" list.

## Limits (say these out loud when you hand it over)

- It's a word list, not a lab. It can't see inside a packet or catch cross-contact
  in a shared kitchen.
- It's biased toward over-blocking, so some fine dishes get rejected.
- For severe allergies, the person with the allergy should read each plan before cooking.
- A 7B model plans decent dinners, not gourmet ones. A bigger model slots in with
  one environment variable.

## Before you hand it over (10 minutes)

- What exactly can't you eat, and how bad is a slip? Any "avoid but not allergic" foods?
- Do you avoid coconut, oats, lecithin or anything else that's a grey area?
- What do you cook on a weeknight? How long do you want to spend?
- What do you eat on repeat? What do you never want to see again?
- Do you want a printed page on the fridge, or something on your phone?

## Post skeleton

- **The person and the problem:** one real scene (the label-reading at the supermarket).
- **Why open mattered:** allergy details stay on the machine, no subscription,
  model swappable, and the safety layer is code you can read and test, not a prompt you hope holds.
- **What broke:** the model's first drafts and what the guard caught (the retry log is great material).
- **What they said:** their reaction, and the first thing they changed.
