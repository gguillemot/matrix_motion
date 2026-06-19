"""Tests unitaires pour l'extraction de contact badge QR (format CSV BreizhCamp)."""
from __future__ import annotations

import unittest

from src.badge_webhook import extract_badge_contact


class TestExtractBadgeContactCSV(unittest.TestCase):
    """Vérifie le parsing du format CSV BreizhCamp."""

    _HEADER = "id,lastname,firstname,email,company,ticketType,twitterAccount,noGoodies,tShirtFitting,tShirtSize"
    _ROW1 = "1,Baratheon,Robert,robert.baratheon@kings-landing.wes,Retired King,Jeudi,@rbaratheon,,Homme,XXXL"
    _ROW2 = "2,STARK,Eddard,eddard.stark@winterfell.wes,Retired Hand of the King,Mercredi,@beheadardstark,1,,"

    def test_csv_with_header_row1(self):
        result = extract_badge_contact(f"{self._HEADER}\n{self._ROW1}")
        self.assertEqual(result, {
            "prenom": "Robert",
            "nom": "Baratheon",
            "email": "robert.baratheon@kings-landing.wes",
        })

    def test_csv_with_header_row2(self):
        result = extract_badge_contact(f"{self._HEADER}\n{self._ROW2}")
        self.assertEqual(result, {
            "prenom": "Eddard",
            "nom": "STARK",
            "email": "eddard.stark@winterfell.wes",
        })

    def test_csv_data_only_no_header(self):
        """QR sans entête : utilise la convention positionnelle."""
        result = extract_badge_contact(self._ROW1)
        self.assertEqual(result, {
            "prenom": "Robert",
            "nom": "Baratheon",
            "email": "robert.baratheon@kings-landing.wes",
        })

    def test_csv_email_lowercased(self):
        row = "3,LANNISTER,Cersei,Cersei.LANNISTER@iron-throne.wes,Queen,,,,,"
        result = extract_badge_contact(row)
        self.assertIsNotNone(result)
        self.assertEqual(result["email"], "cersei.lannister@iron-throne.wes")

    def test_csv_missing_email_returns_none(self):
        """Sans email, le payload ne doit pas être envoyé."""
        row = "4,Snow,Jon,,Night Watch,Mercredi,,,,XL"
        result = extract_badge_contact(row)
        self.assertIsNone(result)

    def test_csv_with_header_multiple_data_rows(self):
        """Plusieurs lignes de données : seule la première valide est renvoyée."""
        csv_block = f"{self._HEADER}\n{self._ROW1}\n{self._ROW2}"
        result = extract_badge_contact(csv_block)
        self.assertIsNotNone(result)
        self.assertEqual(result["email"], "robert.baratheon@kings-landing.wes")

    def test_empty_string_returns_none(self):
        self.assertIsNone(extract_badge_contact(""))

    def test_none_like_empty_returns_none(self):
        self.assertIsNone(extract_badge_contact("   "))



if __name__ == "__main__":
    unittest.main()
