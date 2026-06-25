"""Team name normalization and confederation lookup.

Different sources use different spellings (USA / United States, Korea Republic /
South Korea, Türkiye / Turkey, etc.). This module gives a canonical name used
across the project, and provides confederation lookup for WC 2026 teams.
"""

from __future__ import annotations

# Canonical name -> aliases seen in sources
_ALIASES: dict[str, list[str]] = {
    "United States": ["USA", "US", "USMNT", "United States of America"],
    "South Korea": ["Korea Republic", "Republic of Korea", "Korea"],
    "North Korea": ["Korea DPR", "DPR Korea"],
    "DR Congo": ["Congo DR", "Democratic Republic of the Congo", "Congo-Kinshasa"],
    "Republic of Congo": ["Congo", "Congo-Brazzaville"],
    "Czech Republic": ["Czechia"],
    "Cape Verde": ["Cabo Verde"],
    "Ivory Coast": ["Côte d'Ivoire", "Cote d'Ivoire"],
    "Turkey": ["Türkiye", "Turkiye"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia", "Bosnia & Herzegovina"],
    "North Macedonia": ["Macedonia", "FYR Macedonia"],
    "Iran": ["IR Iran", "Islamic Republic of Iran"],
    "Russia": ["Russian Federation"],
    "Saint Vincent and the Grenadines": ["St Vincent and the Grenadines"],
    "Saint Kitts and Nevis": ["St Kitts and Nevis"],
    "Antigua and Barbuda": ["Antigua & Barbuda"],
    "Trinidad and Tobago": ["Trinidad & Tobago"],
    "São Tomé and Príncipe": ["Sao Tome and Principe"],
    "Equatorial Guinea": ["Eq. Guinea"],
}

# Build flat alias -> canonical map (case-insensitive)
_ALIAS_TO_CANON: dict[str, str] = {}
for canon, aliases in _ALIASES.items():
    _ALIAS_TO_CANON[canon.lower()] = canon
    for a in aliases:
        _ALIAS_TO_CANON[a.lower()] = canon


def normalize(name: str) -> str:
    """Return canonical team name. Unknown names are returned trimmed but unchanged."""
    if not name:
        return name
    cleaned = name.strip()
    return _ALIAS_TO_CANON.get(cleaned.lower(), cleaned)


# Confederation membership (FIFA member federations)
CONFEDERATION: dict[str, str] = {
    # UEFA — Europe
    **dict.fromkeys(
        [
            "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
            "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
            "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
            "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
            "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Kazakhstan",
            "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
            "Malta", "Moldova", "Montenegro", "Netherlands", "North Macedonia",
            "Northern Ireland", "Norway", "Poland", "Portugal", "Romania",
            "Russia", "San Marino", "Scotland", "Serbia", "Slovakia", "Slovenia",
            "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
        ],
        "UEFA",
    ),
    # CONMEBOL — South America
    **dict.fromkeys(
        [
            "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
            "Paraguay", "Peru", "Uruguay", "Venezuela",
        ],
        "CONMEBOL",
    ),
    # CONCACAF — North/Central America/Caribbean
    **dict.fromkeys(
        [
            "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
            "Belize", "Bermuda", "Bonaire", "British Virgin Islands", "Canada",
            "Cayman Islands", "Costa Rica", "Cuba", "Curaçao", "Dominica",
            "Dominican Republic", "El Salvador", "French Guiana", "Grenada",
            "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica",
            "Martinique", "Mexico", "Montserrat", "Nicaragua", "Panama",
            "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia",
            "Saint Martin", "Saint Vincent and the Grenadines", "Sint Maarten",
            "Suriname", "Trinidad and Tobago", "Turks and Caicos Islands",
            "United States", "US Virgin Islands",
        ],
        "CONCACAF",
    ),
    # CAF — Africa
    **dict.fromkeys(
        [
            "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
            "Cameroon", "Cape Verde", "Central African Republic", "Chad",
            "Comoros", "Republic of Congo", "DR Congo", "Djibouti", "Egypt",
            "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
            "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast",
            "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi",
            "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique",
            "Namibia", "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe",
            "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
            "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
            "Zambia", "Zimbabwe",
        ],
        "CAF",
    ),
    # AFC — Asia
    **dict.fromkeys(
        [
            "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan",
            "Brunei", "Cambodia", "China", "Chinese Taipei", "Guam",
            "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan",
            "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau",
            "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
            "North Korea", "Northern Mariana Islands", "Oman", "Pakistan",
            "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
            "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand",
            "Timor-Leste", "Turkmenistan", "United Arab Emirates",
            "Uzbekistan", "Vietnam", "Yemen",
        ],
        "AFC",
    ),
    # OFC — Oceania
    **dict.fromkeys(
        [
            "American Samoa", "Cook Islands", "Fiji", "New Caledonia",
            "New Zealand", "Papua New Guinea", "Samoa", "Solomon Islands",
            "Tahiti", "Tonga", "Vanuatu",
        ],
        "OFC",
    ),
}


def confederation(team: str) -> str:
    """Return confederation code (UEFA, CONMEBOL, ...) or 'UNK' if unknown."""
    return CONFEDERATION.get(normalize(team), "UNK")
