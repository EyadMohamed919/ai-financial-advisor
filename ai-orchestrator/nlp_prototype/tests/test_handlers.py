import datetime as dt
import os
import unittest

os.environ["GEMINI_API_KEY"] = ""  # tests run offline; an empty value also blocks the key in .env

from nlp_proto import handlers  # noqa: E402

TODAY = dt.date(2026, 11, 19)
LEDGER = [
    {"id": 1, "date": "2026-11-19", "type": "expense", "amount": 320, "currency": "EGP", "category": "Food", "label": "بيتزا"},
    {"id": 2, "date": "2026-11-18", "type": "expense", "amount": 1200, "currency": "EGP", "category": "Shopping", "label": "كوتشي"},
    {"id": 3, "date": "2026-11-10", "type": "expense", "amount": 150, "currency": "EGP", "category": "Food", "label": "كشري"},
    {"id": 4, "date": "2026-11-01", "type": "income", "amount": 15000, "currency": "EGP", "category": "Salary", "label": "المرتب"},
    {"id": 5, "date": "2026-11-05", "type": "expense", "amount": 200, "currency": "EGP", "category": "Transport", "label": "بنزين"},
    {"id": 6, "date": "2026-10-20", "type": "expense", "amount": 900, "currency": "EGP", "category": "Food", "label": "عزومة"},
]


class HandlersTest(unittest.TestCase):
    def test_spending_by_category_and_period(self):
        self.assertEqual(handlers.spending_query("صرفت كام على الأكل الشهر ده؟", LEDGER, TODAY)["data"]["total"], 470)
        self.assertEqual(handlers.spending_query("صرفت كام الشهر اللي فات", LEDGER, TODAY)["data"]["total"], 900)
        self.assertEqual(handlers.spending_query("صرفت كام النهارده", LEDGER, TODAY)["data"]["total"], 320)
        self.assertEqual(handlers.spending_query("اكبر مصروف عندي ايه", LEDGER, TODAY)["data"]["top_category"], "Shopping")

    def test_forecast_needs_five_records(self):
        self.assertTrue(handlers.forecast("هصرف كام", LEDGER[:3], TODAY)["data"]["insufficient_data"])
        data = handlers.forecast("هصرف كام الشهر ده", LEDGER, TODAY)["data"]
        self.assertAlmostEqual(data["projected_month"], 1870 / 19 * 30, places=1)

    def test_manage_deletes_latest(self):
        out = handlers.manage("امسح آخر حاجة", LEDGER, TODAY)
        self.assertEqual(out["action"], {"type": "delete", "id": 1})

    def test_advice_without_llm_has_disclaimer(self):
        self.assertIn(handlers.DISCLAIMER, handlers.advice("احوش ازاي", LEDGER, TODAY)["reply"])


if __name__ == "__main__":
    unittest.main()
