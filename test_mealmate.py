"""Run with:  python -m unittest -v   (no model, no network needed)"""
import json
import unittest

from allergens import AllergenGuard
from mealmate import MockClient, build_plan, plan_day, render_html

PROFILE = {"name": "Sam", "allergies": ["peanut", "tree nuts", "shellfish"], "household_size": 2}


def terms(guard, text):
    return {h.term for h in guard.scan(text) if h.level == "block"}


class GuardTests(unittest.TestCase):
    def test_hidden_milk(self):
        g = AllergenGuard(["dairy"])
        for t in ["ghee", "whey protein", "butter", "paneer", "Crème fraîche", "sour cream", "casein"]:
            self.assertTrue(terms(g, t), t)

    def test_hidden_everything_else(self):
        self.assertTrue(terms(AllergenGuard(["sesame"]), "tahini"))
        self.assertTrue(terms(AllergenGuard(["fish"]), "Worcestershire sauce"))
        self.assertTrue(terms(AllergenGuard(["soy"]), "miso paste"))
        self.assertTrue(terms(AllergenGuard(["tree nuts"]), "basil pesto"))
        self.assertTrue(terms(AllergenGuard(["peanut"]), "chicken satay"))
        self.assertTrue(terms(AllergenGuard(["wheat"]), "soy sauce"))
        self.assertTrue(terms(AllergenGuard(["gluten"]), "pearl barley"))
        self.assertTrue(terms(AllergenGuard(["shellfish"]), "oyster sauce"))

    def test_no_false_positives_on_lookalikes(self):
        g = AllergenGuard(["dairy", "egg", "tree nuts", "wheat"])
        for t in ["peanut butter", "coconut milk", "eggplant", "nutmeg", "butternut squash",
                  "water chestnuts", "buckwheat", "rice flour", "oat milk",
                  "cream of tartar", "butter beans", "kale"]:
            self.assertFalse(terms(g, t), t)

    def test_caution_is_not_a_block_unless_strict(self):
        text = "coconut milk"
        self.assertFalse(terms(AllergenGuard(["tree nuts"]), text))
        self.assertTrue(AllergenGuard(["tree nuts"]).scan(text))
        self.assertTrue(terms(AllergenGuard(["tree nuts"], strict=True), text))

    def test_generic_nuts_block_both_kinds(self):
        self.assertTrue(terms(AllergenGuard(["peanut"]), "chopped nuts"))
        self.assertTrue(terms(AllergenGuard(["tree nuts"]), "chopped nuts"))

    def test_unknown_allergy_is_never_dropped(self):
        g = AllergenGuard(["kiwi"])
        self.assertEqual(g.unrecognised, ["kiwi"])
        self.assertTrue(terms(g, "sliced kiwi fruit"))

    def test_also_avoid(self):
        self.assertTrue(terms(AllergenGuard(["egg"], also_avoid=["cilantro"]), "fresh cilantro"))


class ScriptedClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, messages, schema):
        self.calls += 1
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def day(*dishes):
    return json.dumps({"meals": [
        {"slot": s, "name": n, "minutes": 10, "ingredients": [{"item": i, "amount": 1, "unit": ""}], "steps": [st]}
        for (s, n, i, st) in dishes]})


SAFE = day(("breakfast", "Rice porridge", "rice", "Simmer."),
           ("lunch", "Chickpea stew", "chickpeas", "Simmer."),
           ("dinner", "Roast chicken", "chicken", "Roast."))
UNSAFE_INGREDIENT = day(("breakfast", "Rice porridge", "rice", "Simmer."),
                        ("lunch", "Chickpea stew", "chickpeas", "Simmer."),
                        ("dinner", "Prawn curry", "prawns", "Cook."))
UNSAFE_STEP_ONLY = day(("breakfast", "Rice porridge", "rice", "Top with crushed peanuts."),
                       ("lunch", "Chickpea stew", "chickpeas", "Simmer."),
                       ("dinner", "Roast chicken", "chicken", "Roast."))


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.guard = AllergenGuard(PROFILE["allergies"])

    def run_day(self, client):
        return plan_day(client, self.guard, PROFILE, "Monday", [], log=lambda *_: None)

    def test_retries_until_safe(self):
        c = ScriptedClient(UNSAFE_INGREDIENT, SAFE)
        data, _, problems = self.run_day(c)
        self.assertIsNotNone(data)
        self.assertEqual(c.calls, 2)

    def test_steps_are_scanned_not_just_ingredients(self):
        data, _, _ = self.run_day(ScriptedClient(UNSAFE_STEP_ONLY, SAFE))
        self.assertNotIn("peanut", json.dumps(data))

    def test_gives_up_instead_of_returning_unsafe(self):
        c = ScriptedClient(UNSAFE_INGREDIENT)
        data, _, problems = self.run_day(c)
        self.assertIsNone(data)
        self.assertTrue(problems)

    def test_bad_json_and_wrong_slots_are_retried(self):
        c = ScriptedClient("not json", day(("lunch", "x", "rice", "y")), SAFE)
        data, _, _ = self.run_day(c)
        self.assertIsNotNone(data)
        self.assertEqual(c.calls, 3)


class EndToEnd(unittest.TestCase):
    def test_mock_week_is_clean_and_renders(self):
        plan = build_plan(MockClient(), PROFILE, 3, log=lambda *_: None)
        self.assertEqual(len(plan["days"]), 3)
        self.assertFalse(plan["failed"])
        guard = AllergenGuard(PROFILE["allergies"])
        self.assertEqual([], [h for d in plan["days"] for m in d["meals"]
                              for h in guard.scan(json.dumps(m)) if h.level == "block"])
        page = render_html(plan)
        self.assertIn("Shopping list", page)
        self.assertIn("vegetable stock", plan["label_checks"])


if __name__ == "__main__":
    unittest.main()
