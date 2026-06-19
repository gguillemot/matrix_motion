"""Tests unitaires pour l'extraction de contact badge QR (format CSV BreizhCamp)."""
from __future__ import annotations

import unittest

from src.badge_webhook import extract_badge_contact


class TestExtractBadgeContactCSV(unittest.TestCase):
    """Vérifie le parsing du format CSV BreizhCamp.

    Format sans entête (nominal) :
        lastname(0), firstname(1), company(2), email(3)
    """

    # --- Format nominal : sans entête, valeurs entre guillemets ---
    _ROW_LANNISTER = '"LANNISTER","TYRION","Hand of the King","tyrion.lannister@casterly-rock.wes"'
    _ROW_BARATHEON = '"Baratheon","Robert","Retired King","robert.baratheon@kings-landing.wes"'
    _ROW_STARK     = '"STARK","Eddard","Retired Hand of the King","eddard.stark@winterfell.wes"'

    # --- Format rétrocompatible : avec entête ---
    _HEADER = "id,lastname,firstname,email,company,ticketType,twitterAccount,noGoodies,tShirtFitting,tShirtSize"
    _ROW_LEGACY1 = "1,Baratheon,Robert,robert.baratheon@kings-landing.wes,Retired King,Jeudi,@rbaratheon,,Homme,XXXL"
    _ROW_LEGACY2 = "2,STARK,Eddard,eddard.stark@winterfell.wes,Retired Hand of the King,Mercredi,@beheadardstark,1,,"

    # --- Cas nominaux (sans entête) ---

    def test_no_header_lannister(self):
        result = extract_badge_contact(self._ROW_LANNISTER)
        self.assertEqual(result, {
            "prenom": "TYRION",
            "nom": "LANNISTER",
            "email": "tyrion.lannister@casterly-rock.wes",
            "entreprise": "Hand of the King",
        })

    def test_no_header_baratheon(self):
        result = extract_badge_contact(self._ROW_BARATHEON)
        self.assertEqual(result, {
            "prenom": "Robert",
            "nom": "Baratheon",
            "email": "robert.baratheon@kings-landing.wes",
            "entreprise": "Retired King",
        })

    def test_no_header_stark(self):
        result = extract_badge_contact(self._ROW_STARK)
        self.assertEqual(result, {
            "prenom": "Eddard",
            "nom": "STARK",
            "email": "eddard.stark@winterfell.wes",
            "entreprise": "Retired Hand of the King",
        })

    def test_no_header_email_lowercased(self):
        row = '"LANNISTER","Cersei","Queen","Cersei.LANNISTER@iron-throne.wes"'
        result = extract_badge_contact(row)
        self.assertIsNotNone(result)
        self.assertEqual(result["email"], "cersei.lannister@iron-throne.wes")

    def test_no_header_missing_email_returns_none(self):
        """Sans email, le payload ne doit pas être envoyé."""
        row = '"Snow","Jon","Night Watch",""'
        result = extract_badge_contact(row)
        self.assertIsNone(result)

    # --- Rétrocompatibilité : avec entête ---

    def test_csv_with_header_row1(self):
        result = extract_badge_contact(f"{self._HEADER}\n{self._ROW_LEGACY1}")
        self.assertEqual(result, {
            "prenom": "Robert",
            "nom": "Baratheon",
            "email": "robert.baratheon@kings-landing.wes",
            "entreprise": "Retired King",
        })

    def test_csv_with_header_row2(self):
        result = extract_badge_contact(f"{self._HEADER}\n{self._ROW_LEGACY2}")
        self.assertEqual(result, {
            "prenom": "Eddard",
            "nom": "STARK",
            "email": "eddard.stark@winterfell.wes",
            "entreprise": "Retired Hand of the King",
        })

    def test_csv_with_header_multiple_data_rows(self):
        """Plusieurs lignes de données : seule la première valide est renvoyée."""
        csv_block = f"{self._HEADER}\n{self._ROW_LEGACY1}\n{self._ROW_LEGACY2}"
        result = extract_badge_contact(csv_block)
        self.assertIsNotNone(result)
        self.assertEqual(result["email"], "robert.baratheon@kings-landing.wes")

    # --- Cas limites ---

    def test_empty_string_returns_none(self):
        self.assertIsNone(extract_badge_contact(""))

    def test_none_like_empty_returns_none(self):
        self.assertIsNone(extract_badge_contact("   "))



if __name__ == "__main__":
    unittest.main()
