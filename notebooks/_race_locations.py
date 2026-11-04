"""
Race location mapping — structured metadata approach.

Rule: every race is assigned the country that hosts it, and the host country is assigned a
continent by _common.CONTINENT_MAP, the same mapping used for athlete origin. A race and an
athlete from the same country therefore always share a continent (Israel, the Gulf states and
Turkey = Asia; Mexico, Central America and the Caribbean = North America; US states and
territories = North America), whichever source the race comes from:
1. Official source: subevent_id -> parent_slug (all_subevents.csv) -> host country
   (SLUG_TO_COUNTRY, 174 series mapped by hand). The two World Championship slugs are left
   unmapped on purpose so that the year-by-year host fallback below applies to them.
2. Secondary source (aggregator): race_id -> race name (race_metadata.csv) -> series, i.e. the
   race name without its year -> host country (CC_SERIES_TO_COUNTRY, mapped by hand). The World
   Championships are mapped to their host by year.

Race location of secondary-source races. The `region` field of race_metadata.csv is not used
for the mapping: it is IRONMAN's commercial region, not a geographic continent, and it carries
coding errors (70.3 Dubai 2016-2019 and 2022 and 70.3 Tiberias 2021-2022 coded Africa; IRONMAN
Israel 2022 coded Europe; 70.3 Costa Rica 2017-2019 and 70.3 Riviera Nayarit 2024-2025 coded
South America; 70.3 Saipan 2017 coded Asia; and, among races that are not in the merged file,
70.3 Victoria (Canada) coded Australia, the 2025 women's IRONMAN World Championship in Kona
(race 2203) coded Europe, 70.3 Bahrain, Oman, Salalah and Muscat races coded Africa, and Mexican,
Central American and Caribbean 70.3 races coded South America). The race location of these races
is therefore the series -> host country mapping, and the official series follow the same rule
(im703-turkey: Turkey is Asia in CONTINENT_MAP; im703-san-juan: San Juan, Argentina, not Puerto
Rico). The next version of the companion dataset will carry the same correction. The region
field is read only to report how many starters the host-country rule places on another continent
than the region field (athlete_vs_venue.txt section 1).

Both metadata files are part of the companion data paper's collection and are public in the
repository https://github.com/Nova-O2/ironman-data (data/collection/all_subevents.csv and
data/collection/race_metadata.csv; the public all_subevents.csv carries one extra column,
`reconstructed`, which is not read). The default paths point at the local raw-data folder;
IRONMAN_DNF_SUBEVENTS_PATH and IRONMAN_DNF_RACE_METADATA_PATH override them. A missing file
raises FileNotFoundError, an aggregator series without a host country raises ValueError, and
assign_race_continent() fails when any record is left without a race location (coverage is 100%
of starters in dataset v4.0.0).

Assertions (build_race_location_table): every host country is a key of CONTINENT_MAP. The
continent of a race is CONTINENT_MAP of its host country, so a country cannot receive two
continents (that agreement holds by construction and is not tested). What is tested is the
agreement between the two sources: every series matched across them by its normalised name
(event type + place, e.g. 'him:coeur-dalene' for im703-coeur-dalene and "Ironman 70.3 Coeur
d'Alene"), and every series across its years, receives a single host COUNTRY. The World
Championships are the only exception: they move between hosts by year.

No race location is inferred from keywords in event_name.
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import CONTINENT_MAP

DATA_RAW = Path(__file__).parent.parent.parent.parent / "_data_raw"
SUBEVENTS_PATH = Path(os.environ.get("IRONMAN_DNF_SUBEVENTS_PATH",
                                     DATA_RAW / "ironman_official" / "all_subevents.csv")).resolve()
CC_META_PATH = Path(os.environ.get("IRONMAN_DNF_RACE_METADATA_PATH",
                                   DATA_RAW / "_legacy" / "coachcox" / "race_metadata.csv")).resolve()

# === Official: parent_slug -> host country ===
# 124 series of the official index of dataset v3 and 50 series that entered it with dataset v4, each
# mapped by hand to its host country. 'im-world-championship-kona' and 'im703-world-championship' are
# deliberately absent (see _fallback_event_name). Comments give the continent that CONTINENT_MAP
# assigns.
SLUG_TO_COUNTRY = {
    # Full distance — North America
    "im-california": "United States", "im-chattanooga": "United States",
    "im-florida": "United States", "im-lake-placid": "United States",
    "im-maryland": "United States", "im-texas": "United States",
    "im-wisconsin": "United States", "im-cozumel": "Mexico",
    # Full distance — South America
    "im-brazil": "Brazil",
    # Full distance — Europe
    "im-austria": "Austria", "im-barcelona": "Spain",
    "im-copenhagen": "Denmark", "im-france": "France",
    "im-frankfurt": "Germany", "im-hamburg": "Germany",
    "im-lanzarote": "Spain", "im-leeds": "United Kingdom",
    "im-switzerland": "Switzerland", "im-tallinn": "Estonia",
    "im-vichy": "France", "im-wales": "United Kingdom",
    # Full distance — Asia
    "im-gurye": "South Korea", "im-philippines": "Philippines", "im-taiwan": "Taiwan",
    # Full distance — Oceania
    "im-australia": "Australia", "im-cairns": "Australia",
    "im-new-zealand": "New Zealand", "im-western-australia": "Australia",
    # Full distance — Africa
    "im-south-africa": "South Africa",
    # 70.3 — North America (Hawaii: a US state, North America with the United States)
    "im703-acapulco": "Mexico", "im703-augusta": "United States",
    "im703-boulder": "United States", "im703-calgary": "Canada",
    "im703-campeche": "Mexico", "im703-cap-cana": "Dominican Republic",
    "im703-chattanooga": "United States", "im703-coeur-dalene": "United States",
    "im703-cozumel": "Mexico", "im703-des-moines": "United States",
    "im703-eagleman": "United States", "im703-florida": "United States",
    "im703-gulf-coast": "United States", "im703-hawaii": "United States",
    "im703-louisville": "United States", "im703-maine": "United States",
    "im703-michigan": "United States",
    "im703-mont-tremblant": "Canada", "im703-monterrey": "Mexico",
    "im703-muncie": "United States", "im703-muskoka": "Canada",
    "im703-musselman": "United States", "im703-new-york": "United States",
    "im703-north-carolina": "United States", "im703-oceanside": "United States",
    "im703-ohio": "United States", "im703-oregon": "United States",
    "im703-panama": "Panama", "im703-puerto-rico": "Puerto Rico",   # named "IRONMAN 70.3 San Juan" until 2016
    "im703-santa-cruz": "United States",
    "im703-texas": "United States", "im703-victoria": "Canada",
    "im703-waco": "United States", "im703-western-massachusetts": "United States",
    "im703-wisconsin": "United States",
    # 70.3 — South America
    "im703-aracaju-sergipe": "Brazil", "im703-buenos-aires": "Argentina",
    "im703-cartagena": "Colombia", "im703-ecuador": "Ecuador",
    "im703-encarnacion": "Paraguay", "im703-florianopolis": "Brazil",
    "im703-fortaleza": "Brazil", "im703-peru": "Peru",
    "im703-pucon": "Chile", "im703-punta-del-este": "Uruguay",
    "im703-rio-de-janeiro": "Brazil",
    "im703-san-juan": "Argentina",   # San Juan, Argentina (editions from 2022; not the Puerto Rico race)
    "im703-valdivia": "Chile",
    # 70.3 — Europe
    "im703-aix-en-provence": "France", "im703-belgrade": "Serbia",
    "im703-bolton": "United Kingdom", "im703-duisburg": "Germany",
    "im703-elsinore": "Denmark", "im703-erkner": "Germany",
    "im703-gdynia": "Poland", "im703-knokke-heist": "Belgium",
    "im703-kraichgau": "Germany", "im703-mallorca": "Spain",
    "im703-marathon": "Greece",   # Marathon, Greece (not Marathon FL)
    "im703-nice": "France", "im703-poznan": "Poland",
    "im703-swansea": "United Kingdom", "im703-switzerland": "Switzerland",
    "im703-tallinn": "Estonia",
    "im703-valencia": "Spain", "im703-venice-jesolo": "Italy",
    "im703-vichy": "France", "im703-warsaw": "Poland",
    "im703-westfriesland": "Netherlands", "im703-weymouth": "United Kingdom",
    # 70.3 — Asia (Turkey: Asia in CONTINENT_MAP, as for an athlete from Turkey)
    "im703-turkey": "Turkey",
    "im703-bahrain": "Bahrain", "im703-colombo": "Sri Lanka",
    "im703-davao-philippines": "Philippines", "im703-desaru-coast": "Malaysia",
    "im703-goa": "India", "im703-goseong": "South Korea",
    "im703-japan": "Japan", "im703-kenting": "Taiwan",
    "im703-oman": "Oman", "im703-phu-quoc": "Vietnam",
    "im703-puerto-princesa": "Philippines", "im703-vietnam": "Vietnam",
    # 70.3 — Oceania
    "im703-cairns": "Australia", "im703-geelong": "Australia",
    "im703-melbourne": "Australia", "im703-new-zealand": "New Zealand",
    "im703-port-macquarie": "Australia", "im703-sunshine-coast": "Australia",
    "im703-western-australia": "Australia", "im703-western-sydney": "Australia",
    # 70.3 — Africa
    "im703-agadir": "Morocco", "im703-durban": "South Africa",
    "im703-mossel-bay": "South Africa", "im703-rwanda": "Rwanda",
    # === Series that entered the official index with dataset v4 ===
    # Full distance — North America
    "im-canada-ottawa": "Canada", "im-jacksonville": "United States",
    # Full distance — Europe
    "im-cascais": "Portugal", "im-emilia-romagna": "Italy", "im-kalmar": "Sweden",
    "im-les-sables-dolonne": "France", "im-tours": "France", "im-vitoria": "Spain",
    # Full distance — Asia
    "im-malaysia": "Malaysia", "im-south-hokkaido": "Japan", "im-vietnam": "Vietnam",
    # 70.3 — North America
    "im703-boise": "United States", "im703-dallas-little-elm": "United States",
    "im703-la-quinta": "United States", "im703-los-cabos": "Mexico",
    "im703-northern-california": "United States", "im703-omaha": "United States",
    "im703-pennsylvania": "United States", "im703-rockford-illinois": "United States",
    "im703-ruidoso-new-mexico": "United States", "im703-san-salvador": "El Salvador",
    "im703-washington-tri-cities": "United States",
    # 70.3 — South America
    "im703-barranquilla": "Colombia", "im703-brasilia": "Brazil",
    "im703-curitiba": "Brazil", "im703-puerto-varas": "Chile",
    "im703-sao-paulo": "Brazil",
    # 70.3 — Europe (Azerbaijan: Europe in CONTINENT_MAP)
    "im703-alghero": "Italy", "im703-baku": "Azerbaijan", "im703-cascais": "Portugal",
    "im703-croatia": "Croatia", "im703-emilia-romagna": "Italy",
    "im703-hradec-kralove": "Czech Republic", "im703-jonkoping": "Sweden", "im703-krakow": "Poland",
    "im703-leipzig": "Germany", "im703-les-sables-dolonne": "France",
    "im703-luxembourg": "Luxembourg", "im703-tours": "France", "im703-versailles": "France",
    "im703-vitoria": "Spain", "im703-zell-am-see": "Austria",
    # 70.3 — Asia
    "im703-cebu-philippines": "Philippines", "im703-dali": "China", "im703-dhofar": "Oman",
    "im703-hengqin": "China", "im703-langkawi": "Malaysia", "im703-shanghai-chongming": "China",
    "im703-subic-bay-philippines": "Philippines",
    # 70.3 — Africa
    "im703-nelson-mandela-bay": "South Africa",
}

# === Secondary source (aggregator): series (race name without the year) -> host country ===
# All 334 series of race_metadata.csv: 331 here, plus the three World Championship series, which
# are mapped by year (_world_championship_host). The host of each series was checked against the
# official index (same series) and, for the races that are not in it, against the nationality mix
# of the field. Saipan is the Northern Mariana Islands, a US territory in Oceania; it follows
# CONTINENT_MAP, which places the Northern Mariana Islands (with Guam and American Samoa) in North
# America. The 70.3 Asia-Pacific Championship of 2010-2011 is taken as the Laguna Phuket race
# (Thailand), the one whose field it matches (70.3 Laguna Phuket 2012); every candidate host is in
# Asia, so the continent does not depend on it.
_CC_SERIES_BY_COUNTRY = {
    "United States": [
        "Ironman 70.3 Arizona", "Ironman 70.3 Atlantic City", "Ironman 70.3 Atlantic City Aqaubike",
        "Ironman 70.3 Augusta", "Ironman 70.3 Austin", "Ironman 70.3 Boise", "Ironman 70.3 Boulder",
        "Ironman 70.3 Branson", "Ironman 70.3 Buffalo Springs Lake", "Ironman 70.3 California",
        "Ironman 70.3 Chattanooga", "Ironman 70.3 Coeur d'Alene", "Ironman 70.3 Connecticut",
        "Ironman 70.3 Dallas", "Ironman 70.3 Des Moines", "Ironman 70.3 Eagleman", "Ironman 70.3 Florida",
        "Ironman 70.3 Gulf Coast", "Ironman 70.3 Happy Valley", "Ironman 70.3 Hawaii",
        "Ironman 70.3 Indian Wells La Quinta", "Ironman 70.3 Kansas", "Ironman 70.3 La Quinta",
        "Ironman 70.3 Lake Placid", "Ironman 70.3 Lake Stevens", "Ironman 70.3 Lake Tahoe",
        "Ironman 70.3 Louisville", "Ironman 70.3 Lubbock", "Ironman 70.3 Maine", "Ironman 70.3 Miami",
        "Ironman 70.3 Michigan", "Ironman 70.3 Mooseman", "Ironman 70.3 Muncie", "Ironman 70.3 Musselman",
        "Ironman 70.3 Musselman (No Swim Split)", "Ironman 70.3 New Orleans", "Ironman 70.3 New York",
        "Ironman 70.3 North Carolina", "Ironman 70.3 Oceanside", "Ironman 70.3 Ohio", "Ironman 70.3 Omaha",
        "Ironman 70.3 Oregon", "Ironman 70.3 Pocono Mountains", "Ironman 70.3 Princeton",
        "Ironman 70.3 Racine", "Ironman 70.3 Raleigh", "Ironman 70.3 Rhode Island", "Ironman 70.3 Rockford",
        "Ironman 70.3 Santa Cruz", "Ironman 70.3 Santa Rosa", "Ironman 70.3 Silverman",
        "Ironman 70.3 St. George", "Ironman 70.3 Steelhead", "Ironman 70.3 Superfrog",
        "Ironman 70.3 Syracuse", "Ironman 70.3 Texas", "Ironman 70.3 Timberman",
        "Ironman 70.3 Traverse City", "Ironman 70.3 Vineman", "Ironman 70.3 Virginia",
        "Ironman 70.3 Virginia's Blue Ridge", "Ironman 70.3 Waco", "Ironman 70.3 Washington",
        "Ironman 70.3 Washington Tri-Cities", "Ironman 70.3 Western Massachusetts", "Ironman 70.3 Wisconsin",
        "Ironman Alaska", "Ironman Arizona", "Ironman Boulder", "Ironman California", "Ironman Chattanooga",
        "Ironman Coeur d'Alene", "Ironman Des Moines", "Ironman Florida", "Ironman Indiana",
        "Ironman Jacksonville", "Ironman Lake Placid", "Ironman Lake Tahoe", "Ironman Louisville",
        "Ironman Maryland", "Ironman New York", "Ironman North Carolina", "Ironman Santa Rosa",
        "Ironman St George", "Ironman Texas", "Ironman Tulsa", "Ironman Vineman", "Ironman Waco",
        "Ironman Wisconsin",
    ],
    "Canada": [
        "Ironman 70.3 Calgary", "Ironman 70.3 Calgary (Full Bike)", "Ironman 70.3 Calgary (Partial Bike)",
        "Ironman 70.3 Canada", "Ironman 70.3 Mont-Tremblant", "Ironman 70.3 Muskoka",
        "Ironman 70.3 Victoria",   # Victoria, British Columbia (region field: "Australia")
        "Ironman Canada", "Ironman Mont-Tremblant", "Ironman Muskoka", "Ironman Ottawa",
    ],
    "Mexico": [
        "Ironman 70.3 Acapulco", "Ironman 70.3 Campeche", "Ironman 70.3 Cancun", "Ironman 70.3 Cozumel",
        "Ironman 70.3 Los Cabos", "Ironman 70.3 Monterrey", "Ironman 70.3 Riviera Nayarit",
        "Ironman Cozumel", "Ironman Los Cabos",
    ],
    "Puerto Rico": ["Ironman 70.3 Puerto Rico"],
    "U.S. Virgin Islands": ["Ironman 70.3 St Croix"],
    "Northern Mariana Islands": ["Ironman 70.3 Saipan"],
    "Dominican Republic": ["Ironman 70.3 Cap Cana"],
    "Costa Rica": ["Ironman 70.3 Costa Rica"],
    "Panama": ["Ironman 70.3 Panama"],
    "El Salvador": ["Ironman 70.3 San Salvador"],
    "Brazil": [
        "Ironman 70.3 Alagoas", "Ironman 70.3 Aracaju-Sergipe", "Ironman 70.3 Brasilia", "Ironman 70.3 Brazil",
        "Ironman 70.3 Florianopolis", "Ironman 70.3 Fortaleza", "Ironman 70.3 Foz do Iguacu",
        "Ironman 70.3 Maceio", "Ironman 70.3 Palmas", "Ironman 70.3 Rio de Janeiro", "Ironman 70.3 São Paulo",
        "Ironman Brazil", "Ironman Fortaleza",
    ],
    "Argentina": [
        "Ironman 70.3 Bariloche", "Ironman 70.3 Buenos Aires", "Ironman 70.3 Mar del Plata",
        "Ironman 70.3 San Juan",   # from 2022; the earlier races of that name are in Puerto Rico (CC_SERIES_ALIAS_UNTIL)
        "Ironman 70.3 South American Championship", "Ironman Argentina",
    ],
    "Chile": ["Ironman 70.3 Coquimbo", "Ironman 70.3 Pucon", "Ironman 70.3 Puerto Varas", "Ironman 70.3 Valdivia"],
    "Colombia": ["Ironman 70.3 Cartagena", "Ironman 70.3 San Andres"],
    "Ecuador": ["Ironman 70.3 Ecuador"],
    "Peru": ["Ironman 70.3 Peru"],
    "Paraguay": ["Ironman 70.3 Encarnacion", "Ironman 70.3 Paraguay"],
    "Uruguay": ["Ironman 70.3 Punta del Este"],
    "United Kingdom": [
        "Challenge Weymouth", "Ironman 70.3 Bolton", "Ironman 70.3 Edinburgh", "Ironman 70.3 Staffordshire",
        "Ironman 70.3 Swansea", "Ironman 70.3 UK", "Ironman 70.3 Weymouth", "Ironman Leeds", "Ironman UK",
        "Ironman Wales", "Ironman Weymouth",
    ],
    "Ireland": [
        "Ironman 70.3 Dublin", "Ironman 70.3 Dun Laoghaire", "Ironman 70.3 Galway", "Ironman 70.3 Ireland",
        "Ironman Ireland",
    ],
    "Germany": [
        "Ironman 70.3 Berlin", "Ironman 70.3 Dresden", "Ironman 70.3 Duisburg", "Ironman 70.3 Erkner",
        "Ironman 70.3 European Championship",   # Wiesbaden, 2011-2014
        "Ironman 70.3 European Championship Wiesbaden", "Ironman 70.3 Kraichgau", "Ironman 70.3 Ruegen",
        "Ironman 70.3 Wiesbaden", "Ironman Frankfurt", "Ironman Hamburg", "Ironman Regensburg",
    ],
    "France": [
        "Ironman 70.3 Aix-en-Provence", "Ironman 70.3 Les Sables d'Olonne", "Ironman 70.3 Les Sables d'Olonne-Vendée",
        "Ironman 70.3 Nice", "Ironman 70.3 Tours Metropole - Loire Valley", "Ironman 70.3 Vichy", "Ironman France",
        "Ironman Les Sables", "Ironman Pays d'Aix-en-Provence", "Ironman Tours Metropole", "Ironman Vichy",
    ],
    "Italy": [
        "Ironman 70.3 Alghero", "Ironman 70.3 Italy Emilia-Romagna", "Ironman 70.3 Sardegna",
        "Ironman 70.3 Venice-Jesolo", "Ironman Italy",
    ],
    "Spain": [
        "Ironman 70.3 Barcelona", "Ironman 70.3 Lanzarote", "Ironman 70.3 Mallorca", "Ironman 70.3 Marbella",
        "Ironman 70.3 Valencia", "Ironman Barcelona", "Ironman Lanzarote", "Ironman Mallorca",
        "Ironman Vitoria-Gasteiz",
    ],
    "Portugal": ["Ironman 70.3 Portugal - Cascais", "Ironman Portugal - Cascais"],
    "Andorra": ["Ironman 70.3 Andorra"],
    "Switzerland": ["Ironman 70.3 Switzerland", "Ironman Switzerland"],
    "Austria": ["Ironman 70.3 Graz", "Ironman 70.3 St Polten", "Ironman 70.3 Zell am See-Kaprun", "Ironman Austria"],
    "Denmark": [
        "Challenge Copenhagen", "Ironman 70.3 Aarhus", "Ironman 70.3 Elsinore", "Ironman 70.3 Kronborg",
        "Ironman Copenhagen",
    ],
    "Sweden": ["Ironman 70.3 Jönköping", "Ironman Sweden"],
    "Norway": ["Ironman 70.3 Haugesund", "Ironman 70.3 Sandnes", "Ironman Norway"],
    "Finland": ["Ironman 70.3 Finland", "Ironman Finland"],
    "Estonia": ["Ironman 70.3 Otepaa", "Ironman 70.3 Tallinn", "Ironman Tallinn"],
    "Poland": [
        "Ironman 70.3 Gdynia", "Ironman 70.3 Krakow", "Ironman 70.3 Poznan", "Ironman 70.3 Warsaw",
        "Ironman Gdynia",
    ],
    "Czech Republic": ["Ironman 70.3 Hradec Králové"],
    "Hungary": ["Ironman 70.3 Budapest"],
    "Croatia": ["Ironman 70.3 Porec", "Ironman 70.3 Pula"],
    "Slovenia": ["Ironman 70.3 Slovenian Istria"],
    "Serbia": ["Ironman 70.3 Belgrade"],
    "Greece": ["Ironman 70.3 Greece", "Ironman 70.3 Marathon"],
    "Netherlands": ["Ironman 70.3 Westfriesland", "Ironman Maastricht-Limburg"],
    "Belgium": ["Ironman 70.3 Knokke-Heist"],
    "Luxembourg": ["Ironman 70.3 Luxembourg", "Ironman 70.3 Luxembourg-Region Moselle"],
    "Russia": ["Ironman 70.3 St Petersburg"],
    "Turkey": ["Ironman 70.3 Turkey"],
    "Israel": ["Ironman 70.3 Tiberias", "Ironman Israel"],
    "United Arab Emirates": ["Ironman 70.3 Dubai"],
    "Bahrain": ["Ironman 70.3 Bahrain"],
    "Oman": ["Ironman 70.3 Muscat", "Ironman 70.3 Oman", "Ironman 70.3 Salalah"],
    "Kazakhstan": ["Ironman 70.3 Astana", "Ironman Kazakhstan"],
    "India": ["Ironman 70.3 Goa"],
    "Sri Lanka": ["Ironman 70.3 Colombo", "Ironman 70.3 Sri Lanka"],
    "China": [
        "Ironman 70.3 China", "Ironman 70.3 Hefei", "Ironman 70.3 Liuzhou", "Ironman 70.3 Qujing",
        "Ironman 70.3 Shanghai", "Ironman 70.3 Shanghai Chongming", "Ironman 70.3 Xi'an",
        "Ironman 70.3 Xiamen", "Ironman China",
    ],
    "Taiwan": ["Ironman 70.3 Kenting", "Ironman 70.3 Taitung", "Ironman Penghu", "Ironman Taiwan"],
    "Japan": ["Ironman 70.3 Japan", "Ironman Japan"],
    "South Korea": [
        "Ironman 70.3 Busan", "Ironman 70.3 Chungju", "Ironman 70.3 Goseong", "Ironman 70.3 Gurye Korea",
        "Ironman 70.3 Incheon Korea", "Ironman 70.3 Jeju", "Ironman 70.3 Korea", "Ironman Gurye",
    ],
    "Philippines": [
        "Ironman 70.3 Davao Philippines", "Ironman 70.3 Lapu Lapu", "Ironman 70.3 Philippines",
        "Ironman 70.3 Puerto Princesa", "Ironman 70.3 Subic Bay", "Ironman 70.3 Subic Bay Philippines",
        "Ironman Philippines",
    ],
    "Vietnam": ["Ironman 70.3 Da Nang", "Ironman 70.3 Phu Quoc", "Ironman 70.3 Vietnam", "Ironman Vietnam"],
    "Thailand": ["Ironman 70.3 Asia-Pacific", "Ironman 70.3 Bangsaen", "Ironman 70.3 Laguna Phuket",
                 "Ironman 70.3 Thailand"],
    "Malaysia": ["Ironman 70.3 Desaru Coast", "Ironman 70.3 Langkawi", "Ironman 70.3 Putrajaya", "Ironman Malaysia"],
    "Singapore": ["Ironman 70.3 Singapore"],
    "Indonesia": ["Ironman 70.3 Bintan", "Ironman 70.3 Lombok"],
    "Australia": [
        "Ironman 70.3 Ballarat", "Ironman 70.3 Busselton", "Ironman 70.3 Cairns", "Ironman 70.3 Canberra",
        "Ironman 70.3 Geelong", "Ironman 70.3 Mandurah", "Ironman 70.3 Melbourne", "Ironman 70.3 Port Macquarie",
        "Ironman 70.3 Shepparton", "Ironman 70.3 Sunshine Coast", "Ironman 70.3 Western Australia",
        "Ironman 70.3 Western Sydney", "Ironman 70.3 Yeppoon", "Ironman Australia", "Ironman Cairns",
        "Ironman Melbourne", "Ironman Western Australia",
    ],
    "New Zealand": ["Ironman 70.3 Auckland", "Ironman 70.3 New Zealand", "Ironman 70.3 Taupo", "Ironman New Zealand"],
    "South Africa": [
        "Ironman 70.3 Durban", "Ironman 70.3 Mossel Bay", "Ironman 70.3 Nelson Mandela Bay",
        "Ironman 70.3 South Africa", "Ironman South Africa",
    ],
    "Morocco": ["Ironman 70.3 Agadir", "Ironman 70.3 Marrakech", "Ironman 70.3 Tangier"],
    "Egypt": ["Ironman 70.3 Egypt"],
    "Rwanda": ["Ironman 70.3 Rwanda"],
}
CC_SERIES_TO_COUNTRY = {s: c for c, names in _CC_SERIES_BY_COUNTRY.items() for s in names}
assert len(CC_SERIES_TO_COUNTRY) == sum(len(v) for v in _CC_SERIES_BY_COUNTRY.values()), "series listed twice"

# A series name the aggregator reuses for another venue: its "Ironman 70.3 San Juan" races up to
# 2016 are the San Juan (Puerto Rico) race that the official index files under im703-puerto-rico
# (named "IRONMAN 70.3 San Juan" until 2016); from 2022 the name is the race in San Juan, Argentina.
CC_SERIES_ALIAS_UNTIL = {"Ironman 70.3 San Juan": (2016, "Ironman 70.3 Puerto Rico")}

# === Aggregator region codes — read only to report the correction, never to map ===
# The region field uses the literal code "NA" for North America, which pandas reads as missing
# by default; the file is therefore read with keep_default_na=False.
CC_REGION_TO_CONTINENT = {
    "NA": "North America", "Euro": "Europe", "Australia": "Oceania",
    "Asia": "Asia", "SA": "South America", "Africa": "Africa",
}

# === Official events not in all_subevents.csv ===
# Fallback by event_name for official event_ids absent from the official index. In dataset
# v4.0.0 every official event_id is in the index (the fallback is kept for other versions);
# the World Championship events reach _fallback_event_name because their two series slugs are
# not in SLUG_TO_COUNTRY, and are mapped to their host by year.
UNMAPPED_EVENT_COUNTRY = {
    "Emilia Romagna": "Italy", "Italy Emilia Romagna": "Italy", "Emilia Romagna Bike-Run": "Italy",
    "Kalmar-Sweden": "Sweden", "Kalmar": "Sweden",
    "Portugal-Cascais": "Portugal", "Cascais": "Portugal",
}

# === World Championship hosts by year ===
# Sources: Wikipedia (Ironman World Championship, Ironman 70.3 World Championship).
# IRONMAN: Kailua-Kona (Hawaii) every year except 2021 (St George, Utah, held in May 2022);
# 2020 cancelled. In 2023-2025 the men's and women's races were held apart, one in Nice in
# September and the other in Kona in October.
WC_IM_HOST = {**{y: "United States" for y in range(2003, 2020)},
              2021: "United States", 2022: "United States", 2026: "United States"}
WC_IM_SPLIT_HOST = {
    2023: {"Men": "France", "Women": "United States"},
    2024: {"Men": "United States", "Women": "France"},
    2025: {"Men": "France", "Women": "United States"},
}
WC_IM_SPLIT_HOST_BY_MONTH = {9: "France", 10: "United States"}   # the aggregator's race names carry no sex
# IRONMAN 70.3: Clearwater FL 2006-2010, Henderson NV 2011-2013, Mont-Tremblant 2014,
# Zell am See 2015, Mooloolaba 2016, Chattanooga 2017, Nelson Mandela Bay 2018, Nice 2019,
# St George 2021-2022, Lahti 2023, Taupo 2024, Marbella 2025, Nice 2026; 2020 cancelled.
WC_703_HOST = {**{y: "United States" for y in range(2006, 2014)},
               2014: "Canada", 2015: "Austria", 2016: "Australia", 2017: "United States",
               2018: "South Africa", 2019: "France", 2021: "United States", 2022: "United States",
               2023: "Finland", 2024: "New Zealand", 2025: "Spain", 2026: "France"}


def _world_championship_host(race_type: str, year: int, women: bool | None = None,
                             month: int | None = None) -> str:
    """Host country of a World Championship race. For the full distance in 2023-2025 the sex of
    the race (official event names) or, failing that, its month (aggregator) picks the host."""
    if race_type == "him":
        if year not in WC_703_HOST:
            raise KeyError(f"IRONMAN 70.3 World Championship host unknown for {year}")
        return WC_703_HOST[year]
    if year in WC_IM_SPLIT_HOST:
        if women is not None:
            return WC_IM_SPLIT_HOST[year]["Women" if women else "Men"]
        if month not in WC_IM_SPLIT_HOST_BY_MONTH:
            raise ValueError(f"IRONMAN World Championship {year}: race month {month} is neither September nor October")
        return WC_IM_SPLIT_HOST_BY_MONTH[month]
    if year not in WC_IM_HOST:
        raise KeyError(f"IRONMAN World Championship host unknown for {year}")
    return WC_IM_HOST[year]


def _read_metadata(path: Path, env_var: str, columns: list[str], **kwargs) -> pd.DataFrame:
    """Read one race-location metadata file; fail loudly when it is absent or incomplete."""
    if not path.exists():
        raise FileNotFoundError(
            f"race-location metadata file not found: {path} ({path.name}; public copy in "
            f"https://github.com/Nova-O2/ironman-data, data/collection/{path.name}; set {env_var} "
            f"to its location)")
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"{path}: required column(s) missing: {missing}")
    return pd.read_csv(path, usecols=columns, **kwargs)[columns]   # column order as requested


def load_subevents() -> pd.DataFrame:
    """Official index of race editions (subevent_id -> parent_slug)."""
    return _read_metadata(SUBEVENTS_PATH, "IRONMAN_DNF_SUBEVENTS_PATH", ["parent_slug", "subevent_id"])


def load_cc_metadata() -> pd.DataFrame:
    """Aggregator race metadata (race_id, race name, event type, year, date, region code read literally)."""
    return _read_metadata(CC_META_PATH, "IRONMAN_DNF_RACE_METADATA_PATH",
                          ["race_id", "race_name", "race_type", "race_year", "race_date", "region"],
                          keep_default_na=False, dtype={"race_name": str, "race_type": str,
                                                        "race_date": str, "region": str})


def series_name(race_name: str) -> str:
    """Aggregator race name without its year ('Ironman 70.3 Dubai 2017' -> 'Ironman 70.3 Dubai')."""
    return re.sub(r"\s+", " ", re.sub(r"\b(?:19|20)\d{2}\b", "", str(race_name))).strip()


def series_key(race_type: str, place: str) -> str:
    """Cross-source series key: event type + place in lower-case ASCII, words joined by '-', the
    form of the official slugs (('him', "Coeur d'Alene") -> 'him:coeur-dalene')."""
    s = unicodedata.normalize("NFKD", place).encode("ascii", "ignore").decode().lower().replace("'", "")
    return f"{race_type}:{re.sub(r'[^a-z0-9]+', '-', s).strip('-')}"


def _cc_place(series: str) -> str:
    for prefix in ("Ironman 70.3 ", "Ironman "):
        if series.startswith(prefix):
            return series[len(prefix):]
    return series


def build_race_location_table() -> pd.DataFrame:
    """One row per race of the two metadata files: source, event_id, series, series_key, year,
    host country, continent (CONTINENT_MAP of the host country), World Championship flag and, for
    the aggregator, the continent of its region code (kept only for the comparison). The official
    World Championship editions are not listed: their slugs are unmapped and they are placed by
    _fallback_event_name. Raises when a series has no host, a host country is not in
    CONTINENT_MAP, or a non-World Championship series gets more than one host country across sources
    or years (a country always gets one continent: the continent is CONTINENT_MAP of the country)."""
    rows = []
    for slug, sid in load_subevents().itertuples(index=False):
        if slug in SLUG_TO_COUNTRY:
            rt = "him" if slug.startswith("im703-") else "im"
            place = slug.split("-", 1)[1]
            rows.append({"source": "official", "event_id": str(sid), "series": slug,
                         "series_key": series_key(rt, place), "year": None,
                         "country": SLUG_TO_COUNTRY[slug], "is_world_championship": False,
                         "region_continent": None})

    cc = load_cc_metadata()
    unknown_regions = sorted(set(cc["region"]) - set(CC_REGION_TO_CONTINENT))
    if unknown_regions:
        raise ValueError(f"{CC_META_PATH}: unknown region code(s): {unknown_regions}")
    unmapped = set()
    for rid, name, rt, year, date, region in cc.itertuples(index=False):
        year = int(year)
        series = series_name(name)
        is_wc = "World Championship" in series
        if is_wc:
            month = int(date[5:7]) if re.match(r"^\d{4}-\d{2}", date or "") else None
            country = _world_championship_host(rt, year, month=month)
        else:
            alias = CC_SERIES_ALIAS_UNTIL.get(series)
            if alias and year <= alias[0]:
                series = alias[1]
            country = CC_SERIES_TO_COUNTRY.get(series)
            if country is None:
                unmapped.add(series)
                continue
        rows.append({"source": "coachcox", "event_id": str(int(rid)), "series": series,
                     "series_key": series_key(rt, _cc_place(series)), "year": year,
                     "country": country, "is_world_championship": is_wc,
                     "region_continent": CC_REGION_TO_CONTINENT[region]})
    if unmapped:
        raise ValueError(f"{CC_META_PATH}: series without a host country in CC_SERIES_TO_COUNTRY: {sorted(unmapped)}")

    table = pd.DataFrame(rows)
    table["year"] = table["year"].astype("Int64")
    not_in_map = sorted(set(table["country"]) - set(CONTINENT_MAP))
    if not_in_map:
        raise ValueError(f"host countries not in CONTINENT_MAP: {not_in_map}")
    table["continent"] = table["country"].map(CONTINENT_MAP)

    # Single host country per series, across sources and years (series matched across the two
    # sources by series_key). A single continent per country needs no test: continent = CONTINENT_MAP[country].
    ser = table[~table["is_world_championship"]]
    per_series = ser.groupby("series_key")["country"].nunique()
    bad = per_series[per_series > 1].index.tolist()
    if bad:
        detail = ser[ser["series_key"].isin(bad)].groupby(["series_key", "source", "country"]).size()
        raise AssertionError(f"series with more than one host country across sources or years:\n{detail}")
    dup = table["event_id"].duplicated(keep=False)
    if dup.any():
        raise AssertionError(f"event_id listed twice: {table.loc[dup, 'event_id'].unique()[:5].tolist()}")
    return table


def build_race_continent_map() -> dict[str, str]:
    """event_id (str) -> race continent, from both metadata sources (build_race_location_table)."""
    t = build_race_location_table()
    return dict(zip(t["event_id"], t["continent"]))


def _fallback_event_name(event_name: str) -> str | None:
    """Host country of an official event that is not in SLUG_TO_COUNTRY, from its event_name.

    World Championships are mapped to their host by year (and, for the full distance in
    2023-2025, by the sex named in the event name).
    """
    if pd.isna(event_name):
        return None
    name = str(event_name)

    # Non-WC keywords first
    for kw, country in UNMAPPED_EVENT_COUNTRY.items():
        if kw.lower() in name.lower():
            return country

    if "World Championship" not in name:
        return None
    year_match = re.search(r"\b((?:19|20)\d{2})\b", name)   # e.g. "2023 IRONMAN World Championship - Women"
    if not year_match:
        raise ValueError(f"World Championship event without a year in its name: {name!r}")
    return _world_championship_host("him" if "70.3" in name else "im", int(year_match.group(1)),
                                    women="women" in name.lower())


def assign_race_continent(df: pd.DataFrame, require_full_coverage: bool = True) -> pd.DataFrame:
    """Add race_continent column to dataframe using structured metadata.

    Requires 'event_id' and 'event_name' columns in df.
    Strategy:
    1. Primary: event_id -> host country (official series slug or aggregator series) -> continent
    2. Fallback: event_name (World Championships by year; official events absent from the index)
    With require_full_coverage (default), any record left without a race location raises
    AssertionError (coverage is 100% of starters in dataset v4.0.0); otherwise such records
    are labelled "Unknown".
    """
    if "event_id" not in df.columns:
        raise ValueError("event_id column required")

    race_map = build_race_continent_map()
    df["race_continent"] = df["event_id"].astype(str).map(race_map)

    # Fallback for unmapped events using event_name (host country -> CONTINENT_MAP)
    if "event_name" in df.columns:
        mask = df["race_continent"].isna()
        if mask.any():
            names = df.loc[mask, "event_name"].astype(str)
            hosts = {n: _fallback_event_name(n) for n in names.unique()}
            df.loc[mask, "race_continent"] = names.map(lambda n: CONTINENT_MAP.get(hosts[n]) if hosts[n] else None)

    unmapped = df["race_continent"].isna()
    if require_full_coverage and unmapped.any():
        top = (df.loc[unmapped, "event_name"].astype(str).value_counts().head(5).to_dict()
               if "event_name" in df.columns else {})
        raise AssertionError(f"race location missing for {int(unmapped.sum()):,} of {len(df):,} records "
                             f"(top events: {top}); extend SLUG_TO_COUNTRY or the fallbacks")
    df["race_continent"] = df["race_continent"].fillna("Unknown").astype("category")
    return df


def region_field_continent_map() -> dict[str, str]:
    """Aggregator race_id (str) -> continent of its region code, used only to report how many starters the
    host-country rule places on another continent than the region field."""
    t = build_race_location_table()
    t = t[t["source"] == "coachcox"]
    return dict(zip(t["event_id"], t["region_continent"]))


if __name__ == "__main__":
    table = build_race_location_table()
    print(f"Races mapped from the metadata: {len(table):,} "
          f"({(table['source'] == 'official').sum():,} official, {(table['source'] == 'coachcox').sum():,} aggregator); "
          f"host countries: {table['country'].nunique()}")
    print(table.groupby(["source", "continent"]).size().to_string())

    # Test coverage on the analysed population (excluded race-years removed by load_dataset)
    from _common import load_dataset
    df = load_dataset(["event_id", "event_name", "source", "finish_status"])
    starters = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()
    starters = assign_race_continent(starters, require_full_coverage=False)

    mapped = (starters["race_continent"] != "Unknown").sum()
    total = len(starters)
    print(f"\nCoverage on starters: {mapped:,} / {total:,} ({mapped/total*100:.1f}%)")
    print(f"By source:")
    for src in ["official", "coachcox"]:
        sub = starters[starters["source"] == src]
        m = (sub["race_continent"] != "Unknown").sum()
        print(f"  {src}: {m:,} / {len(sub):,} ({m/len(sub)*100:.1f}%)")
    print(f"\nBy continent:")
    print(starters["race_continent"].value_counts().to_string())
