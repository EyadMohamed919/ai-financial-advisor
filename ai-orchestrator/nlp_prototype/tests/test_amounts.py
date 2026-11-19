import unittest

from nlp_proto.amounts import extract_amount, parse_number_text


class ExtractAmountTest(unittest.TestCase):
    cases = [
        ("انا اشتريت كوتشي نايك من مول مصر ب 1200 جنيه", 1200, "EGP"),
        ("قبضت المرتب ١٥٠٠٠", 15000, "EGP"),
        ("دفعت ميتين جنيه بنزين", 200, "EGP"),
        ("دفعت مصاريف الكورس ألف وخمسمية", 1500, "EGP"),
        ("دفعت ٢ ونص ألف جنيه في طقم حلل", 2500, "EGP"),
        ("طيرت باكو ونص في لبس شتوي", 1500, "EGP"),
        ("اشتريت 3 تيشيرتات ب 900", 900, "EGP"),
        ("shareet 3 shai w 2ahwa b 75", 75, "EGP"),
        ("dafa3t 60 koshary", 60, "EGP"),
        ("masarif el gam3a 9.5k", 9500, "EGP"),
        ("Card ****4821 charged EGP 285.00 at TALABAT on 03/10/2026. Avl bal EGP 9,420.50", 285, "EGP"),
        ("InstaPay: You sent EGP 500.00 to MOHAMED A. Ref 88213", 500, "EGP"),
        ("Card ****4821 charged USD 9.99 at SPOTIFY", 9.99, "USD"),
        ("دفعت ٢٠ دولار اشتراك ChatGPT", 20, "USD"),
    ]

    def test_amounts(self):
        for text, amount, currency in self.cases:
            with self.subTest(text=text):
                got = extract_amount(text)
                self.assertAlmostEqual(got["amount"], amount)
                self.assertEqual(got["currency"], currency)

    def test_no_amount(self):
        for text in ("واحد صاحبي عزمني", "عامل ايه النهارده؟"):
            with self.subTest(text=text):
                self.assertIsNone(extract_amount(text)["amount"])

    def test_parse_span(self):
        for span, value in (("١٢٠٠", 1200), ("1,280.00", 1280), ("تلتمية وخمسين", 350), ("نص أرنب", 500000), ("تلت تلاف", 3000)):
            with self.subTest(span=span):
                self.assertAlmostEqual(parse_number_text(span), value)


if __name__ == "__main__":
    unittest.main()
