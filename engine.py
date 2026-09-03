from dataclasses import dataclass
from typing import Optional
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import folium
from shapely.geometry import Point
from pyproj import Geod, CRS


GEOD = Geod(ellps="WGS84")

@dataclass
class StoreInput:
    """
    Generic store input object.

    Uses attribute access so it is directly compatible
    with the existing OSM retrieval engine.
    """

    city: str
    latitude: float
    longitude: float
    auto_tops: int
    pole_kiosks: int
    no_parking_boards: int

ENGINE_CONFIG = {

    "schema_version": "1.0",

    # -----------------------------------------------------
    # Catchment defaults
    # -----------------------------------------------------
    "catchment": {
        "zones": {
            "Z1": {"min_m": 0, "max_m": 300},
            "Z2": {"min_m": 300, "max_m": 750},
            "Z3": {"min_m": 750, "max_m": 1500},
            "Z4": {"min_m": 1500, "max_m": 3000},
        }
    },

    # -----------------------------------------------------
    # Geographic data
    # -----------------------------------------------------
    "geospatial": {
        "default_search_radius_m": 3000,
    },

    # -----------------------------------------------------
    # Distance relevance
    #
    # These are V1 planning defaults.
    # They are configurable and not empirically validated.
    # -----------------------------------------------------
    "distance_weights": {
        "Z1": 1.00,
        "Z2": 0.70,
        "Z3": 0.40,
        "Z4": 0.15,
    },

    # -----------------------------------------------------
    # Media roles
    # -----------------------------------------------------
    "media": {
        "auto_tops": "movement_corridor",
        "pole_kiosks": "high_value_nodes",
        "no_parking_boards": "local_coverage",
    },

    # -----------------------------------------------------
    # Allocation
    # -----------------------------------------------------
    "allocation": {
        "rounding_method": "largest_remainder",
        "exact_reconciliation": True,
    },

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------
    "output": {
        "include_raw_evidence": True,
        "include_confidence": True,
        "include_methodology": True,
    }
}

ALLOCATION_CONFIG = {

    "minimum_units": {
        "auto_tops": 1,
        "pole_kiosks": 1,
        "no_parking_boards": 1,
    },

    "minimum_opportunity_index": {
        "auto_tops": 10.0,
        "pole_kiosks": 10.0,
        "no_parking_boards": 10.0,
    },

    "maximum_target_share": {

        # Auto Tops are spread across corridors.
        "auto_tops": 0.40,

        # Kiosks are point/area placements, so retain
        # some concentration control.
        "pole_kiosks": 0.30,

        # CONSOLIDATION FIX: boards now allocate across real
        # candidate clusters (see prepare/cluster_no_parking_
        # candidates), not just the 4 catchment rings, so a hard
        # cap no longer starves the plan the way it would have
        # against only 4 candidates. Capped at 35% per cluster as
        # a concentration safeguard -- the original 1.00 (no cap)
        # let the whole package land in one ring/cluster with
        # nothing checking whether that many physical sites
        # actually exist there.
        "no_parking_boards": 0.35,
    },

    "candidate_limits": {
        "auto_tops": 25,
        "pole_kiosks": 30,
        "no_parking_boards": 50,
    },

    "rounding_method": "largest_remainder",
}

DENSITY_CONFIG = {

    # Categories treated as commercial ecosystem signals
    "commercial_ecosystem": {
        "commercial": 1.00,
        "market": 1.25,
        "beauty_bpc": 1.50,
        "commercial_support": 0.75,
        "food": 0.50,
    },

    # Categories treated as movement signals
    "movement_ecosystem": {
        "transit_major": 1.50,
        "transit_secondary": 0.75,
    },

    # Categories representing reasons for people to visit/
    # spend time in an area
    "trip_generator_ecosystem": {
        "healthcare": 1.00,
        "education": 0.75,
        "financial": 0.50,
        "community_leisure": 0.50,
        "accommodation": 0.40,
        "public_service": 0.40,
        "trip_generator": 1.00,
    },

    # Generic evidence treatment.
    # Named features are currently treated as stronger
    # evidence than unnamed features.
    #
    # These are model parameters, not empirical estimates.
    "evidence_weights": {
        "named": 1.00,
        "unnamed": 0.50,
    },
}

MEDIA_OPPORTUNITY_CONFIG = {

    "auto_tops": {
        "segment_movement": 0.50,
        "segment_commercial": 0.20,
        "segment_trip_generators": 0.15,
        "road_structure": 0.15,
    },

    "pole_kiosks": {
        "node_connectivity": 0.25,
        "node_commercial": 0.35,
        "node_movement": 0.20,
        "node_trip_generators": 0.20,
    },

    "no_parking_boards": {
        "commercial": 0.55,
        "movement": 0.20,
        "trip_generators": 0.15,
        "local_feature_density": 0.10,
    },
}

FEATURE_TAXONOMY = {

    # =====================================================
    # PLACE / DESTINATION CATEGORIES
    # =====================================================

    "commercial": {
        "shop": {
            "supermarket",
            "convenience",
            "department_store",
            "general",
            "clothes",
            "shoes",
            "jewelry",
            "electronics",
            "computer",
            "mobile_phone",
            "furniture",
            "hardware",
            "hardware_store",
            "stationery",
            "books",
            "bicycle",
            "car",
            "motorcycle",
            "car_parts",
            "doityourself",
            "variety_store",
        }
    },

    "beauty_bpc": {
        "shop": {
            "beauty",
            "cosmetics",
            "hairdresser",
            "perfumery",
        }
    },

    "market": {
        "amenity": {
            "marketplace",
        },
        "shop": {
            "market",
        }
    },

    "healthcare": {
        "amenity": {
            "hospital",
            "clinic",
            "doctors",
            "dentist",
            "pharmacy",
            "veterinary",
        },
        "shop": {
            "chemist",
        }
    },

    "education": {
        "amenity": {
            "school",
            "college",
            "university",
            "kindergarten",
            "library",
            "music_school",
        },
        "office": {
            "educational_institution",
        }
    },

    "financial": {
        "amenity": {
            "bank",
            "atm",
        }
    },

    "food": {
        "amenity": {
            "restaurant",
            "cafe",
            "fast_food",
            "food_court",
            "ice_cream",
        },
        "shop": {
            "bakery",
            "pastry",
            "confectionery",
            "beverages",
        }
    },

    "accommodation": {
        "tourism": {
            "hotel",
            "guest_house",
            "hostel",
            "motel",
        }
    },

    "community_leisure": {
        "amenity": {
            "community_centre",
            "events_venue",
            "social_centre",
        },
        "leisure": {
            "park",
            "playground",
            "pitch",
            "sports_centre",
            "fitness_centre",
            "common",
        }
    },

    "public_service": {
        "amenity": {
            "police",
            "fire_station",
            "post_office",
        },
        "office": {
            "government",
        }
    },

    "trip_generator": {
        "amenity": {
            "fuel",
            "parking",
            # CONSOLIDATION FIX: "ferry_terminal" removed from here --
            # it was also listed under transit_major.amenity below,
            # and classify_osm_feature checks transit_major first, so
            # this entry could never actually be selected. Kept under
            # transit_major only, which is where it always resolved to
            # anyway.
        }
    },

    # =====================================================
    # TRANSIT
    # =====================================================

    "transit_major": {
        "railway": {
            "station",
            "halt",
        },
        "public_transport": {
            "station",
        },
        "amenity": {
            "bus_station",
            "ferry_terminal",
        }
    },

    "transit_secondary": {
        "railway": {
            "stop",
            "subway_entrance",
        },
        "public_transport": {
            "stop_position",
        },
        "highway": {
            "bus_stop",
        },
        "amenity": {
            "taxi",
        }
    },

    # =====================================================
    # ROAD NETWORK
    # =====================================================

    "road": {
        "highway": {
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "residential",
            "living_street",
            "pedestrian",
            "unclassified",
            "service",
        }
    }
}

NODE_OPPORTUNITY_WEIGHTS = {
    "connectivity": 0.30,
    "commercial": 0.30,
    "movement": 0.25,
    "trip_generators": 0.15,
}

OSM_QUERY_TAGS = {
    "shop": True,
    "amenity": True,
    "office": True,
    "tourism": True,
    "leisure": True,
    "public_transport": True,
    "railway": True,
    "highway": True,
}

RAW_OSM_COLUMNS = [
    "element_type",
    "osmid",
    "name",
    "shop",
    "amenity",
    "office",
    "tourism",
    "leisure",
    "public_transport",
    "railway",
    "highway",
    "geometry",
]

EXPECTED_FEATURE_COLUMNS = [
    "name",
    "shop",
    "amenity",
    "office",
    "tourism",
    "leisure",
    "public_transport",
    "railway",
    "highway",
    "geometry",
    "feature_category",
]

PLACE_CATEGORIES = {
    "commercial",
    "beauty_bpc",
    "market",
    "healthcare",
    "education",
    "financial",
    "food",
    "accommodation",
    "community_leisure",
    "public_service",
    "trip_generator",
}

TRANSIT_CATEGORIES = {
    "transit_major",
    "transit_secondary",
}

REQUIRED_SPATIAL_COLUMNS = {
    "feature_category",
    "evidence_status",
    "distance_m",
    "catchment_zone",
    "geometry",
}

REQUIRED_DENSITY_COLUMNS = {
    "zone",
    "feature_count",
    "area_km2",
    "commercial_signal",
    "movement_signal",
    "trip_generator_signal",
    "commercial_density",
    "movement_density",
    "trip_generator_density",
}

ROAD_CLASS_WEIGHTS = {
    "motorway": 1.00,
    "trunk": 0.95,
    "primary": 0.90,
    "secondary": 0.80,
    "tertiary": 0.65,
    "residential": 0.35,
    "living_street": 0.25,
    "pedestrian": 0.20,
    "service": 0.15,
    "unclassified": 0.20,
}

SEGMENT_OPPORTUNITY_WEIGHTS = {
    "road_structure": 0.25,
    "commercial": 0.30,
    "movement": 0.30,
    "trip_generators": 0.15,
}

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",         # default
    "https://overpass.kumi.systems/api",   # independent operator -- best test of
                                            # whether it's overpass-api.de specifically
    "https://lz4.overpass-api.de/api",     # load-balanced variant of the default
]

def _fetch_with_retry(fn, *args, max_attempts=4, backoff_base_s=2.0, **kwargs):
    """
    CONSOLIDATION FIX (v2): the original notebook called the Overpass
    API (via osmnx) with no error handling anywhere. This retries
    with exponential backoff, and -- new in this version -- falls
    through to alternate public Overpass mirrors if the default one
    is unreachable entirely (connection refused, not just slow),
    rather than only ever retrying the same host.

    Not independently verified that every mirror below is currently
    up -- these are well-known public Overpass mirrors, but their
    availability can change. If all three fail, that's a strong
    signal the problem is on this end (e.g. hosting network
    restrictions), not a one-off Overpass outage.
    """
    last_error = None
    for mirror in OVERPASS_MIRRORS:
        ox.settings.overpass_url = mirror
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                wait_s = backoff_base_s * (2 ** (attempt - 1))
                print(
                    f"  OSM request via {mirror} failed "
                    f"(attempt {attempt}/{max_attempts}): {exc!r} -- "
                    f"retrying in {wait_s:.0f}s"
                )
                time.sleep(wait_s)
        print(f"  {mirror} exhausted -- trying next mirror if any remain...")
    raise RuntimeError(
        f"OSM request failed after trying all {len(OVERPASS_MIRRORS)} "
        f"Overpass mirrors: {last_error!r}"
    ) from last_error


def classify_osm_feature(row, taxonomy=FEATURE_TAXONOMY):
    """
    Classify an OSM feature using the generic taxonomy.

    Priority:
    1. Major transit
    2. Secondary transit
    3. Road
    4. Place categories

    Returns a standardized category.
    """

    attributes = {
        "shop": str(row.get("shop", "")).lower(),
        "amenity": str(row.get("amenity", "")).lower(),
        "office": str(row.get("office", "")).lower(),
        "tourism": str(row.get("tourism", "")).lower(),
        "leisure": str(row.get("leisure", "")).lower(),
        "public_transport": str(
            row.get("public_transport", "")
        ).lower(),
        "railway": str(row.get("railway", "")).lower(),
        "highway": str(row.get("highway", "")).lower(),
    }

    priority_order = [
        "transit_major",
        "transit_secondary",
        "road",
        "market",
        "beauty_bpc",
        "healthcare",
        "education",
        "financial",
        "food",
        "accommodation",
        "community_leisure",
        "public_service",
        "trip_generator",
        "commercial",
    ]

    for category in priority_order:

        category_rules = taxonomy.get(category, {})

        for attribute, valid_values in category_rules.items():

            value = attributes.get(attribute, "")

            if value in valid_values:
                return category

    return "unclassified"

def fetch_osm_features(store, radius_m=None, tags=None):

    if radius_m is None:
        radius_m = ENGINE_CONFIG["geospatial"]["default_search_radius_m"]

    if tags is None:
        tags = OSM_QUERY_TAGS

    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    print(
        f"Fetching OSM features for {store.city} "
        f"within {radius_m:,}m..."
    )

    # CONSOLIDATION FIX: retry with backoff instead of a bare call.
    features = _fetch_with_retry(
        ox.features_from_point,
        center_point=(store.latitude, store.longitude),
        tags=tags,
        dist=radius_m
    )

    if features.empty:
        print("No OSM features returned.")
        return features

    features = features.reset_index()

    print(f"Raw OSM features returned: {len(features):,}")

    return features


def standardise_osm_features(gdf):

    if gdf.empty:
        return gdf.copy()

    result = gdf.copy()

    for column in RAW_OSM_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    standard_columns = [
        column
        for column in RAW_OSM_COLUMNS
        if column in result.columns
    ]

    remaining_columns = [
        column
        for column in result.columns
        if column not in standard_columns
    ]

    return result[
        standard_columns + remaining_columns
    ].copy()

def prepare_osm_features(store):

    raw_features = fetch_osm_features(store)

    if raw_features.empty:
        return raw_features

    features = standardise_osm_features(raw_features)

    features["feature_category"] = features.apply(
        classify_osm_feature,
        axis=1
    )

    return features

def validate_feature_output(gdf):

    missing = [
        column
        for column in EXPECTED_FEATURE_COLUMNS
        if column not in gdf.columns
    ]

    if missing:
        raise AssertionError(
            f"Missing required columns: {missing}"
        )

    return True

def build_store_osm_layer(
    store
):
    """
    Retrieve and prepare OSM features exactly once.

    Reproduces the existing prepare_osm_features()
    transformation while retaining the raw dataset.
    """

    # -----------------------------------------------------
    # One and only one retrieval
    # -----------------------------------------------------

    raw_features = fetch_osm_features(
        store
    )

    if raw_features is None:
        raise RuntimeError(
            "fetch_osm_features() returned None."
        )

    if raw_features.empty:
        raise ValueError(
            f"No OSM features found for {store.city}."
        )

    # -----------------------------------------------------
    # Existing standardisation
    # -----------------------------------------------------

    prepared = standardise_osm_features(
        raw_features
    )

    # -----------------------------------------------------
    # Existing generic classification
    # -----------------------------------------------------

    prepared[
        "feature_category"
    ] = prepared.apply(
        classify_osm_feature,
        axis=1
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    validate_feature_output(
        prepared
    )

    return raw_features, prepared

def add_spatial_attributes(gdf, store, config=None):
    """
    Add generic spatial attributes to any OSM feature set.

    Adds:
        distance_m
        catchment_zone

    Uses only the supplied store coordinates
    and engine configuration.
    """

    if gdf.empty:
        return gdf.copy()

    if config is None:
        config = ENGINE_CONFIG

    zones = config["catchment"]["zones"]

    result = gdf.copy()

    distances = []

    for geometry in result.geometry:

        if geometry is None or geometry.is_empty:
            distances.append(float("nan"))
            continue

        if geometry.geom_type == "Point":
            point = geometry
        else:
            point = geometry.representative_point()

        lon = point.x
        lat = point.y

        _, _, distance = GEOD.inv(
            store.longitude,
            store.latitude,
            lon,
            lat
        )

        distances.append(float(distance))

    result["distance_m"] = distances

    def get_zone(distance):

        if pd.isna(distance):
            return "UNKNOWN"

        for zone_name, bounds in zones.items():

            minimum = bounds["min_m"]
            maximum = bounds["max_m"]

            if minimum <= distance <= maximum:
                return zone_name

        return "OUTSIDE"

    result["catchment_zone"] = (
        result["distance_m"]
        .apply(get_zone)
    )

    return result

def split_feature_families(gdf):
    """
    Separate the standardized geographic dataset into
    analytical families.

    Places, transit and roads are handled differently
    downstream.
    """

    return {
        "places": gdf[
            gdf["feature_category"].isin(
                PLACE_CATEGORIES
            )
        ].copy(),

        "transit": gdf[
            gdf["feature_category"].isin(
                TRANSIT_CATEGORIES
            )
        ].copy(),

        "roads": gdf[
            gdf["feature_category"] == "road"
        ].copy(),

        "unclassified": gdf[
            gdf["feature_category"] == "unclassified"
        ].copy(),
    }

def enrich_feature_families(gdf, store):
    """
    Generic spatial enrichment pipeline:

    classification
    → feature family
    → evidence status
    → distance
    → catchment zone
    """

    result = gdf.copy()

    result = add_evidence_status(result)

    families = split_feature_families(result)

    for family_name, family_gdf in families.items():

        if not family_gdf.empty:

            families[family_name] = (
                add_spatial_attributes(
                    family_gdf,
                    store
                )
            )

    return families

def add_evidence_status(gdf):
    """
    Add a generic evidence-status field.

    named   = feature has a usable OSM name
    unnamed = feature has no usable OSM name

    This does NOT mean unnamed features are invalid.
    It records identification status only.
    """

    if gdf.empty:
        result = gdf.copy()
        result["evidence_status"] = pd.Series(
            dtype="object"
        )
        return result

    result = gdf.copy()

    if "name" not in result.columns:
        result["evidence_status"] = "unnamed"
        return result

    names = (
        result["name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["evidence_status"] = np.where(
        names != "",
        "named",
        "unnamed"
    )

    return result

def validate_enriched_features(gdf):
    """
    Validate the minimum spatial evidence contract.
    """

    if gdf.empty:
        return True

    missing = (
        REQUIRED_SPATIAL_COLUMNS
        - set(gdf.columns)
    )

    if missing:
        raise AssertionError(
            f"Missing required columns: {sorted(missing)}"
        )

    distances = gdf["distance_m"].dropna()

    if (distances < 0).any():
        raise AssertionError(
            "Negative distance detected."
        )

    allowed_evidence = {
        "named",
        "unnamed",
    }

    actual_evidence = set(
        gdf["evidence_status"]
        .dropna()
        .unique()
    )

    if not actual_evidence.issubset(
        allowed_evidence
    ):
        raise AssertionError(
            f"Unexpected evidence status: "
            f"{actual_evidence}"
        )

    allowed_zones = {
        "Z1",
        "Z2",
        "Z3",
        "Z4",
        "OUTSIDE",
        "UNKNOWN",
    }

    actual_zones = set(
        gdf["catchment_zone"]
        .dropna()
        .unique()
    )

    if not actual_zones.issubset(
        allowed_zones
    ):
        raise AssertionError(
            f"Unexpected catchment zones: "
            f"{actual_zones}"
        )

    return True

def get_evidence_weight(row, config=DENSITY_CONFIG):
    """
    Return the evidence-confidence multiplier.

    Named features receive stronger identification confidence
    than unnamed features.
    """

    status = row.get(
        "evidence_status",
        "unnamed"
    )

    return config[
        "evidence_weights"
    ].get(
        status,
        0.50
    )

def get_category_weight(
    category,
    ecosystem,
    config=DENSITY_CONFIG
):
    """
    Return the analytical category weight for a given
    ecosystem.

    Unknown categories receive zero contribution.
    """

    return config[
        ecosystem
    ].get(
        category,
        0.0
    )

def add_evidence_contributions(
    gdf,
    config=DENSITY_CONFIG
):
    """
    Add generic contribution columns for:

        commercial
        movement
        trip generation

    No location-specific logic.
    """

    if gdf.empty:
        return gdf.copy()

    result = gdf.copy()

    result["evidence_weight"] = (
        result.apply(
            get_evidence_weight,
            axis=1
        )
    )

    result["commercial_contribution"] = (
        result["feature_category"].apply(
            lambda category:
            get_category_weight(
                category,
                "commercial_ecosystem",
                config
            )
        )
        * result["evidence_weight"]
    )

    result["movement_contribution"] = (
        result["feature_category"].apply(
            lambda category:
            get_category_weight(
                category,
                "movement_ecosystem",
                config
            )
        )
        * result["evidence_weight"]
    )

    result["trip_generator_contribution"] = (
        result["feature_category"].apply(
            lambda category:
            get_category_weight(
                category,
                "trip_generator_ecosystem",
                config
            )
        )
        * result["evidence_weight"]
    )

    return result

def add_distance_relevance(
    gdf,
    config=ENGINE_CONFIG
):
    """
    Add configurable distance relevance to each feature.

    This uses the catchment-zone configuration rather than
    any location-specific information.
    """

    if gdf.empty:
        return gdf.copy()

    result = gdf.copy()

    distance_weights = config[
        "distance_weights"
    ]

    result["distance_weight"] = (
        result["catchment_zone"]
        .map(distance_weights)
        .fillna(0.0)
    )

    return result

def build_spatial_evidence(gdf):
    """
    Combine:

        category relevance
        × evidence confidence
        × distance relevance

    Produces independent evidence signals.

    """

    if gdf.empty:
        return gdf.copy()

    result = add_evidence_contributions(gdf)
    result = add_distance_relevance(result)

    result["commercial_spatial_signal"] = (
        result["commercial_contribution"]
        * result["distance_weight"]
    )

    result["movement_spatial_signal"] = (
        result["movement_contribution"]
        * result["distance_weight"]
    )

    result["trip_generator_spatial_signal"] = (
        result["trip_generator_contribution"]
        * result["distance_weight"]
    )

    return result

def aggregate_zone_density(
    evidence_gdf,
    config=None
):
    """
    Aggregate feature-level spatial evidence into
    catchment-zone level measures.

    Outputs:
        raw feature counts
        commercial signal
        movement signal
        trip-generator signal

    No location-specific assumptions.
    """

    if config is None:
        config = ENGINE_CONFIG

    zones = list(
        config["catchment"]["zones"].keys()
    )

    if evidence_gdf.empty:

        return pd.DataFrame({
            "zone": zones,
            "feature_count": 0,
            "commercial_signal": 0.0,
            "movement_signal": 0.0,
            "trip_generator_signal": 0.0,
        })

    grouped = (
        evidence_gdf
        .groupby("catchment_zone")
        .agg(
            feature_count=(
                "feature_category",
                "count"
            ),
            commercial_signal=(
                "commercial_spatial_signal",
                "sum"
            ),
            movement_signal=(
                "movement_spatial_signal",
                "sum"
            ),
            trip_generator_signal=(
                "trip_generator_spatial_signal",
                "sum"
            ),
        )
        .reset_index()
        .rename(
            columns={
                "catchment_zone": "zone"
            }
        )
    )

    result = pd.DataFrame({
        "zone": zones
    })

    result = result.merge(
        grouped,
        on="zone",
        how="left"
    )

    numeric_columns = [
        "feature_count",
        "commercial_signal",
        "movement_signal",
        "trip_generator_signal"
    ]

    result[numeric_columns] = (
        result[numeric_columns]
        .fillna(0)
    )

    return result

def add_area_normalized_density(
    zone_density,
    config=None
):
    """
    Normalize zone evidence by geographic area.

    Density is expressed as weighted evidence per km².
    """

    if config is None:
        config = ENGINE_CONFIG

    result = zone_density.copy()

    area_records = []

    for zone_name, bounds in (
        config["catchment"]["zones"].items()
    ):

        outer = bounds["max_m"]
        inner = bounds["min_m"]

        area_km2 = (
            np.pi *
            (outer**2 - inner**2)
            / 1_000_000
        )

        area_records.append({
            "zone": zone_name,
            "area_km2": area_km2
        })

    areas = pd.DataFrame(
        area_records
    )

    result = result.merge(
        areas,
        on="zone",
        how="left"
    )

    result["commercial_density"] = (
        result["commercial_signal"]
        / result["area_km2"]
    )

    result["movement_density"] = (
        result["movement_signal"]
        / result["area_km2"]
    )

    result["trip_generator_density"] = (
        result["trip_generator_signal"]
        / result["area_km2"]
    )

    return result

def validate_zone_density(df):

    missing = (
        REQUIRED_DENSITY_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise AssertionError(
            f"Missing density columns: {sorted(missing)}"
        )

    if df["area_km2"].le(0).any():
        raise AssertionError(
            "Zone area must be positive."
        )

    density_columns = [
        "commercial_density",
        "movement_density",
        "trip_generator_density"
    ]

    for column in density_columns:

        if df[column].lt(0).any():
            raise AssertionError(
                f"Negative values in {column}."
            )

    return True

def validate_density_inputs(gdf):

    required = {
        "feature_category",
        "evidence_status",
        "distance_weight",
        "commercial_spatial_signal",
        "movement_spatial_signal",
        "trip_generator_spatial_signal",
        "catchment_zone",
        "geometry",
    }

    missing = required - set(gdf.columns)

    if missing:
        raise AssertionError(
            f"Missing density columns: {sorted(missing)}"
        )

    numeric_columns = [
        "distance_weight",
        "commercial_spatial_signal",
        "movement_spatial_signal",
        "trip_generator_spatial_signal",
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            gdf[column]
        ):
            raise AssertionError(
                f"{column} must be numeric."
            )

        if (gdf[column] < 0).any():
            raise AssertionError(
                f"{column} contains negative values."
            )

    return True

def build_store_catchments(
    store,
    radii=None
):
    """
    Build generic radial catchment zones for one store.

    Default zones:
        Z1 = 0–300m
        Z2 = 300–750m
        Z3 = 750–1500m
        Z4 = 1500–3000m

    No city-specific logic.
    """

    if radii is None:
        radii = {
            "Z1": (0, 300),
            "Z2": (300, 750),
            "Z3": (750, 1500),
            "Z4": (1500, 3000)
        }

    if not radii:
        raise ValueError(
            "At least one catchment zone is required."
        )

    records = []

    for zone, bounds in radii.items():

        inner_radius = float(
            bounds[0]
        )

        outer_radius = float(
            bounds[1]
        )

        if inner_radius < 0:
            raise ValueError(
                f"{zone}: inner radius cannot be negative."
            )

        if outer_radius <= inner_radius:
            raise ValueError(
                f"{zone}: outer radius must exceed "
                "inner radius."
            )

        records.append(
            {
                "zone": zone,
                "inner_radius_m": inner_radius,
                "outer_radius_m": outer_radius,
                "area_km2": (
                    np.pi
                    * (
                        outer_radius ** 2
                        - inner_radius ** 2
                    )
                    / 1_000_000
                )
            }
        )

    return pd.DataFrame(
        records
    )

def validate_catchments(
    catchments
):
    """
    Validate generic catchment-zone geometry inputs.
    """

    required = {
        "zone",
        "inner_radius_m",
        "outer_radius_m",
        "area_km2"
    }

    missing = (
        required
        - set(catchments.columns)
    )

    if missing:
        raise AssertionError(
            "Missing catchment columns: "
            f"{sorted(missing)}"
        )

    if catchments.empty:
        raise AssertionError(
            "Catchment table is empty."
        )

    if (
        catchments["inner_radius_m"]
        < 0
    ).any():
        raise AssertionError(
            "Negative inner radius detected."
        )

    if (
        catchments["outer_radius_m"]
        <= catchments["inner_radius_m"]
    ).any():
        raise AssertionError(
            "Invalid catchment radius ordering."
        )

    if (
        catchments["area_km2"]
        <= 0
    ).any():
        raise AssertionError(
            "Catchment area must be positive."
        )

    if catchments["zone"].duplicated().any():
        raise AssertionError(
            "Duplicate catchment zone detected."
        )

    return True

def assign_features_to_catchments(
    features,
    store,
    catchments
):
    """
    Assign each geographic feature to a generic
    catchment zone using its distance from the store.

    Features beyond the outer catchment are labelled
    OUTSIDE.

    Distance is stored in metres.

    No city-specific logic.
    """

    if features.empty:
        return features.copy()

    result = features.copy()

    required_catchment_columns = {
        "zone",
        "inner_radius_m",
        "outer_radius_m"
    }

    missing = (
        required_catchment_columns
        - set(catchments.columns)
    )

    if missing:
        raise AssertionError(
            "Missing catchment columns: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Store point
    # -----------------------------------------------------

    store_point = gpd.GeoSeries(
        [
            Point(
                store.longitude,
                store.latitude
            )
        ],
        crs="EPSG:4326"
    )

    # -----------------------------------------------------
    # Project into a local metre-based CRS
    # -----------------------------------------------------

    local_crs = create_local_projection(
        store
    )

    features_local = result.to_crs(
        local_crs
    )

    store_local = store_point.to_crs(
        local_crs
    ).iloc[0]

    # -----------------------------------------------------
    # Distance from store in metres
    # -----------------------------------------------------

    result["distance_m"] = (
        features_local.geometry
        .distance(store_local)
    )

    # -----------------------------------------------------
    # Catchment classification
    # -----------------------------------------------------

    def classify_distance(
        distance_m
    ):

        for _, zone in catchments.iterrows():

            if (
                distance_m >= zone["inner_radius_m"]
                and
                distance_m < zone["outer_radius_m"]
            ):
                return zone["zone"]

        return "OUTSIDE"

    result["catchment_zone"] = (
        result["distance_m"]
        .apply(
            classify_distance
        )
    )

    return result

def validate_feature_catchment_assignment(
    features,
    catchments
):
    """
    Validate generic feature-to-catchment assignment.
    """

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required = {
        "distance_m",
        "catchment_zone"
    }

    missing = (
        required
        - set(features.columns)
    )

    if missing:
        raise AssertionError(
            "Missing assignment columns: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Valid zone names
    # -----------------------------------------------------

    valid_zones = set(
        catchments["zone"]
        .astype(str)
    )

    valid_zones.add(
        "OUTSIDE"
    )

    observed_zones = set(
        features[
            "catchment_zone"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    invalid_zones = (
        observed_zones
        - valid_zones
    )

    if invalid_zones:
        raise AssertionError(
            "Invalid catchment zones found: "
            f"{sorted(invalid_zones)}"
        )

    # -----------------------------------------------------
    # Distance validation
    # -----------------------------------------------------

    if (
        features["distance_m"]
        .isna()
        .any()
    ):
        raise AssertionError(
            "Some features have missing distances."
        )

    if (
        features["distance_m"]
        .lt(0)
        .any()
    ):
        raise AssertionError(
            "Negative feature distance detected."
        )

    # -----------------------------------------------------
    # Every feature must have a zone
    # -----------------------------------------------------

    if (
        features["catchment_zone"]
        .isna()
        .any()
    ):
        raise AssertionError(
            "Some features have no catchment assignment."
        )

    # -----------------------------------------------------
    # Boundary sanity check
    # -----------------------------------------------------

    max_radius = (
        catchments[
            "outer_radius_m"
        ]
        .max()
    )

    outside_features = features[
        features["catchment_zone"]
        == "OUTSIDE"
    ]

    inside_features = features[
        features["catchment_zone"]
        != "OUTSIDE"
    ]

    if not outside_features.empty:

        if (
            outside_features["distance_m"]
            < max_radius
        ).any():

            raise AssertionError(
                "Feature labelled OUTSIDE is still "
                "inside the maximum catchment radius."
            )

    if not inside_features.empty:

        if (
            inside_features["distance_m"]
            >= max_radius
        ).any():

            raise AssertionError(
                "Feature inside a catchment exceeds "
                "the maximum catchment radius."
            )

    return True

def enrich_store_features(
    features,
    store
):
    """
    Run the generic feature-family enrichment pipeline.

    Returns:
        {
            "places": GeoDataFrame,
            "transit": GeoDataFrame,
            "roads": GeoDataFrame,
            "unclassified": GeoDataFrame
        }

    No city-specific logic.
    """

    if features is None:
        raise ValueError(
            "features cannot be None."
        )

    if not isinstance(
        features,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "features must be a GeoDataFrame."
        )

    if features.empty:
        raise ValueError(
            f"No prepared features available for "
            f"{store.city}."
        )

    # -----------------------------------------------------
    # Existing generic enrichment function
    # -----------------------------------------------------

    enriched = enrich_feature_families(
        features,
        store
    )

    # -----------------------------------------------------
    # Validate return structure
    # -----------------------------------------------------

    if not isinstance(
        enriched,
        dict
    ):
        raise TypeError(
            "enrich_feature_families() must return "
            "a dictionary of feature families."
        )

    required_families = {
        "places",
        "transit",
        "roads",
        "unclassified"
    }

    missing = (
        required_families
        - set(enriched.keys())
    )

    if missing:
        raise AssertionError(
            "Missing feature families: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Validate every family
    # -----------------------------------------------------

    for family_name in required_families:

        family = enriched[
            family_name
        ]

        if not isinstance(
            family,
            gpd.GeoDataFrame
        ):
            raise TypeError(
                f"Feature family '{family_name}' "
                f"must be a GeoDataFrame."
            )

        validate_enriched_features(
            family
        )

    return enriched

def build_store_analytical_evidence(
    enriched_features,
    store
):
    """
    Convert the enriched feature-family dictionary into
    the generic analytical-evidence GeoDataFrame required
    by downstream scoring and density modules.

    No city-specific logic.
    """

    if not isinstance(
        enriched_features,
        dict
    ):
        raise TypeError(
            "enriched_features must be a dictionary."
        )

    required_families = [
        "places",
        "transit",
        "roads",
        "unclassified"
    ]

    missing = [
        family
        for family in required_families
        if family not in enriched_features
    ]

    if missing:
        raise AssertionError(
            "Missing feature families: "
            f"{missing}"
        )

    # -----------------------------------------------------
    # Combine enriched families
    # -----------------------------------------------------

    family_frames = []

    for family_name in required_families:

        family = enriched_features[
            family_name
        ]

        if not isinstance(
            family,
            gpd.GeoDataFrame
        ):
            raise TypeError(
                f"{family_name} must be a GeoDataFrame."
            )

        if not family.empty:
            family_frames.append(
                family.copy()
            )

    if not family_frames:
        raise ValueError(
            f"No enriched evidence available for "
            f"{store.city}."
        )

    combined = gpd.GeoDataFrame(
        pd.concat(
            family_frames,
            ignore_index=True
        ),
        crs=family_frames[0].crs
    )

    # -----------------------------------------------------
    # Build spatial evidence
    # -----------------------------------------------------

    evidence = build_spatial_evidence(
        combined
    )

    if evidence is None:
        raise RuntimeError(
            "build_spatial_evidence() returned None."
        )

    if not isinstance(
        evidence,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "build_spatial_evidence() must return "
            "a GeoDataFrame."
        )

    # -----------------------------------------------------
    # Add evidence contributions
    # -----------------------------------------------------

    evidence = add_evidence_contributions(
        evidence
    )

    if evidence is None:
        raise RuntimeError(
            "add_evidence_contributions() returned None."
        )

    if not isinstance(
        evidence,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "add_evidence_contributions() must return "
            "a GeoDataFrame."
        )

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    required_evidence_columns = {
        "evidence_status",
        "distance_weight",
        "commercial_spatial_signal",
        "movement_spatial_signal",
        "trip_generator_spatial_signal"
    }

    missing_evidence = (
        required_evidence_columns
        - set(evidence.columns)
    )

    if missing_evidence:

        raise AssertionError(
            "Analytical evidence is missing required "
            f"columns: {sorted(missing_evidence)}"
        )

    return evidence

def build_store_zone_density(
    evidence,
    store,
    config=None
):
    """
    Build generic zone-density intelligence from the
    analytical evidence GeoDataFrame.
    """

    if evidence.empty:
        raise ValueError(
            f"No analytical evidence available for "
            f"{store.city}."
        )

    validate_density_inputs(
        evidence
    )

    zone_density = (
        aggregate_zone_density(
            evidence,
            config=config
        )
    )

    zone_density = (
        add_area_normalized_density(
            zone_density,
            config=config
        )
    )

    validate_zone_density(
        zone_density
    )

    return zone_density

def fetch_road_network(
    store,
    radius_m=None
):
    """
    Retrieve the drivable road network around ANY store.

    Uses only:
        latitude
        longitude
        search radius

    No city-specific logic.
    """

    if radius_m is None:
        radius_m = ENGINE_CONFIG[
            "geospatial"
        ]["default_search_radius_m"]

    if radius_m <= 0:
        raise ValueError(
            "radius_m must be positive."
        )

    print(
        f"Fetching road network for {store.city} "
        f"within {radius_m:,}m..."
    )

    # CONSOLIDATION FIX: retry with backoff instead of a bare call.
    graph = _fetch_with_retry(
        ox.graph_from_point,
        center_point=(store.latitude, store.longitude),
        dist=radius_m,
        network_type="drive",
        simplify=True
    )

    print(
        f"Road network nodes: {len(graph.nodes):,}"
    )

    print(
        f"Road network edges: {len(graph.edges):,}"
    )

    return graph


def network_to_geodataframes(graph):
    """
    Convert an OSMnx graph into standardized node and edge
    GeoDataFrames.
    """

    nodes, edges = ox.graph_to_gdfs(
        graph,
        nodes=True,
        edges=True
    )

    return nodes, edges

def calculate_network_metrics(
    graph,
    nodes,
    edges
):
    """
    Calculate generic network structure metrics.

    These are descriptive network measures.
    They are not footfall estimates.
    """

    metrics = {}

    # -----------------------------------------------------
    # Basic size
    # -----------------------------------------------------

    metrics["node_count"] = len(nodes)
    metrics["edge_count"] = len(edges)

    # -----------------------------------------------------
    # Road length
    # -----------------------------------------------------

    if "length" in edges.columns:

        metrics["total_road_length_km"] = (
            edges["length"].sum() / 1000
        )

    else:

        metrics["total_road_length_km"] = None

    # -----------------------------------------------------
    # Intersection degree
    # -----------------------------------------------------

    degrees = dict(
        graph.degree()
    )

    degree_series = pd.Series(
        degrees,
        dtype="float64"
    )

    metrics["mean_node_degree"] = (
        degree_series.mean()
    )

    metrics["median_node_degree"] = (
        degree_series.median()
    )

    metrics["high_connectivity_nodes"] = (
        (degree_series >= 4).sum()
    )

    return metrics

def get_road_class_weight(highway_value):
    """
    Convert an OSM highway value into a generic
    structural planning weight.

    Handles:
        - strings
        - lists
        - tuples
        - NumPy arrays
        - missing values

    This is a structural road-class signal.
    It is NOT a traffic-volume estimate.
    """

    # -----------------------------------------------------
    # Missing value
    # -----------------------------------------------------

    if highway_value is None:
        return 0.0

    # -----------------------------------------------------
    # List-like OSM values
    # -----------------------------------------------------

    if isinstance(
        highway_value,
        (list, tuple, set, np.ndarray)
    ):

        values = []

        for value in highway_value:

            if value is None:
                continue

            if pd.isna(value):
                continue

            values.append(
                str(value).lower().strip()
            )

        if not values:
            return 0.0

        weights = [
            ROAD_CLASS_WEIGHTS.get(
                value,
                0.0
            )
            for value in values
        ]

        return max(weights)

    # -----------------------------------------------------
    # Scalar missing value
    # -----------------------------------------------------

    try:

        if pd.isna(highway_value):
            return 0.0

    except (TypeError, ValueError):
        pass

    # -----------------------------------------------------
    # Normal scalar value
    # -----------------------------------------------------

    value = (
        str(highway_value)
        .lower()
        .strip()
    )

    return ROAD_CLASS_WEIGHTS.get(
        value,
        0.0
    )

def add_road_class_weight(edges):
    """
    Add generic road-class structural weight.
    """

    result = edges.copy()

    if "highway" in result.columns:

        result["road_class_weight"] = (
            result["highway"]
            .apply(
                get_road_class_weight
            )
        )

    else:

        result["road_class_weight"] = 0.0

    return result

def build_node_intelligence(
    graph,
    nodes
):
    """
    Calculate generic connectivity measures
    for every road-network node.

    These are structural network signals only.
    """

    result = nodes.copy()

    degree_dict = dict(
        graph.degree()
    )

    result["degree"] = (
        result.index.map(
            degree_dict
        ).fillna(0)
    )

    # A higher degree means the node connects
    # to more road segments.
    result["connectivity_score"] = (
        result["degree"]
    )

    return result

def enrich_network_nodes(
    nodes,
    store,
    config=None
):
    """
    Add generic spatial attributes to road-network nodes.

    Adds:
        distance_m
        catchment_zone
        distance_weight

    Works for ANY store.
    """

    if config is None:
        config = ENGINE_CONFIG

    if nodes.empty:
        return nodes.copy()

    result = nodes.copy()

    distances = []

    for geometry in result.geometry:

        if geometry is None or geometry.is_empty:

            distances.append(np.nan)
            continue

        point = (
            geometry
            if geometry.geom_type == "Point"
            else geometry.representative_point()
        )

        _, _, distance = GEOD.inv(
            store.longitude,
            store.latitude,
            point.x,
            point.y
        )

        distances.append(float(distance))

    result["distance_m"] = distances

    zones = config["catchment"]["zones"]

    def assign_zone(distance):

        if pd.isna(distance):
            return "UNKNOWN"

        for zone_name, bounds in zones.items():

            if (
                bounds["min_m"]
                <= distance
                <= bounds["max_m"]
            ):
                return zone_name

        return "OUTSIDE"

    result["catchment_zone"] = (
        result["distance_m"]
        .apply(assign_zone)
    )

    result["distance_weight"] = (
        result["catchment_zone"]
        .map(
            config["distance_weights"]
        )
        .fillna(0.0)
    )

    return result

def calculate_node_opportunity(
    nodes
):
    """
    Create a generic structural opportunity score
    for road-network nodes.

    Current signal:
        connectivity × distance relevance

    This is NOT traffic volume.
    It is a structural prioritisation signal.
    """

    if nodes.empty:
        return nodes.copy()

    result = nodes.copy()

    max_degree = result["degree"].max()

    if max_degree == 0:

        result["connectivity_index"] = 0.0

    else:

        result["connectivity_index"] = (
            result["degree"]
            / max_degree
            * 100
        )

    result["node_opportunity"] = (
        result["connectivity_index"]
        * result["distance_weight"]
    )

    return result

def calculate_node_evidence_proximity(
    nodes,
    evidence,
    store,
    radius_m=250
):
    """
    Calculate analytical evidence surrounding each
    road-network node.

    IMPORTANT:
    All spatial operations are performed in a local
    metre-based CRS.

    Parameters
    ----------
    nodes : GeoDataFrame
        Road-network nodes.

    evidence : GeoDataFrame
        Place + transit analytical evidence.

    store : StoreInput
        Store defining the local projection.

    radius_m : int
        Proximity radius in metres.

    Returns
    -------
    GeoDataFrame
        Nodes enriched with nearby evidence signals.

    No location-specific logic.
    """

    if nodes.empty:
        return nodes.copy()

    if radius_m <= 0:
        raise ValueError(
            "radius_m must be positive."
        )

    result = nodes.copy()

    # -----------------------------------------------------
    # Create a local metric CRS centred on this store.
    # -----------------------------------------------------

    local_crs = create_local_projection(store)

    nodes_local = result.to_crs(local_crs)

    # Evidence should also be measured in metres.
    evidence_local = evidence.to_crs(local_crs)

    # -----------------------------------------------------
    # Initialise outputs.
    # -----------------------------------------------------

    result[
        "nearby_commercial_signal"
    ] = 0.0

    result[
        "nearby_movement_signal"
    ] = 0.0

    result[
        "nearby_trip_generator_signal"
    ] = 0.0

    result[
        "nearby_feature_count"
    ] = 0

    if evidence_local.empty:
        return result

    # -----------------------------------------------------
    # Create 250m buffers in METRES.
    # -----------------------------------------------------

    buffered_nodes = nodes_local[
        ["geometry"]
    ].copy()

    buffered_nodes["node_id"] = (
        buffered_nodes.index
    )

    buffered_nodes["geometry"] = (
        buffered_nodes.geometry
        .buffer(radius_m)
    )

    # -----------------------------------------------------
    # Spatial join.
    # -----------------------------------------------------

    joined = gpd.sjoin(
        evidence_local,
        buffered_nodes[
            ["node_id", "geometry"]
        ],
        predicate="within",
        how="inner"
    )

    if joined.empty:
        return result

    # -----------------------------------------------------
    # Aggregate evidence around each node.
    # -----------------------------------------------------

    grouped = (
        joined
        .groupby("node_id")
        .agg(
            nearby_commercial_signal=(
                "commercial_spatial_signal",
                "sum"
            ),
            nearby_movement_signal=(
                "movement_spatial_signal",
                "sum"
            ),
            nearby_trip_generator_signal=(
                "trip_generator_spatial_signal",
                "sum"
            ),
            nearby_feature_count=(
                "feature_category",
                "count"
            )
        )
    )

    # -----------------------------------------------------
    # Join results back to original nodes.
    # -----------------------------------------------------

    result = result.join(
        grouped,
        how="left",
        rsuffix="_calculated"
    )

    calculated_columns = [
        "nearby_commercial_signal",
        "nearby_movement_signal",
        "nearby_trip_generator_signal",
        "nearby_feature_count"
    ]

    for column in calculated_columns:

        calculated_column = (
            f"{column}_calculated"
        )

        if calculated_column in result.columns:

            result[column] = (
                result[
                    calculated_column
                ]
                .fillna(
                    result[column]
                )
            )

            result.drop(
                columns=[
                    calculated_column
                ],
                inplace=True
            )

    result[
        "nearby_feature_count"
    ] = (
        result[
            "nearby_feature_count"
        ]
        .fillna(0)
        .astype(int)
    )

    return result

def create_local_projection(store):
    """
    Create a local Azimuthal Equidistant projection
    centred on the supplied store.

    This gives us metre-based geometry for local
    spatial operations such as buffers and proximity.

    Works for ANY latitude / longitude.
    """

    return CRS.from_proj4(
        f"+proj=aeqd "
        f"+lat_0={store.latitude} "
        f"+lon_0={store.longitude} "
        f"+datum=WGS84 "
        f"+units=m "
        f"+no_defs"
    )

def min_max_normalize(series):
    """
    Generic 0–100 normalization.

    Used only for relative comparison within
    the current location.
    """

    if len(series) == 0:
        return series

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            100.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )

def calculate_composite_node_opportunity(
    nodes
):
    """
    Combine network structure with nearby activity.

    This is a planning priority signal.

    It is NOT a traffic-volume or footfall estimate.
    """

    if nodes.empty:
        return nodes.copy()

    result = nodes.copy()

    result[
        "nearby_commercial_index"
    ] = min_max_normalize(
        result["nearby_commercial_signal"]
    )

    result[
        "nearby_movement_index"
    ] = min_max_normalize(
        result["nearby_movement_signal"]
    )

    result[
        "nearby_trip_generator_index"
    ] = min_max_normalize(
        result[
            "nearby_trip_generator_signal"
        ]
    )

    result[
        "composite_node_opportunity"
    ] = (

        result["connectivity_index"]
        * NODE_OPPORTUNITY_WEIGHTS[
            "connectivity"
        ]

        +

        result["nearby_commercial_index"]
        * NODE_OPPORTUNITY_WEIGHTS[
            "commercial"
        ]

        +

        result["nearby_movement_index"]
        * NODE_OPPORTUNITY_WEIGHTS[
            "movement"
        ]

        +

        result[
            "nearby_trip_generator_index"
        ]
        * NODE_OPPORTUNITY_WEIGHTS[
            "trip_generators"
        ]
    )

    return result

def validate_contextual_nodes(nodes):

    required = {
        "geometry",
        "distance_m",
        "catchment_zone",
        "degree",
        "connectivity_index",
        "nearby_feature_count",
        "nearby_commercial_index",
        "nearby_movement_index",
        "nearby_trip_generator_index",
        "composite_node_opportunity",
    }

    missing = (
        required - set(nodes.columns)
    )

    if missing:
        raise AssertionError(
            f"Missing columns: {sorted(missing)}"
        )

    non_negative = [
        "degree",
        "distance_m",
        "nearby_feature_count",
        "composite_node_opportunity",
    ]

    for column in non_negative:

        if nodes[column].dropna().lt(0).any():

            raise AssertionError(
                f"Negative values found in {column}"
            )

    return True

def build_store_road_network(
    store,
    radius_m=3000
):
    """
    Build the complete generic road-network layer
    for a StoreInput object.

    Returns:
        graph
        nodes
        edges

    No city-specific logic.
    """

    if store is None:
        raise ValueError(
            "store cannot be None."
        )

    if radius_m <= 0:
        raise ValueError(
            "radius_m must be positive."
        )

    # -----------------------------------------------------
    # Fetch road graph
    # -----------------------------------------------------

    graph = fetch_road_network(
        store=store,
        radius_m=radius_m
    )

    if graph is None:
        raise RuntimeError(
            "fetch_road_network() returned None."
        )

    if not isinstance(
        graph,
        nx.MultiDiGraph
    ):
        raise TypeError(
            "Expected a networkx.MultiDiGraph."
        )

    # -----------------------------------------------------
    # Convert graph to GeoDataFrames
    # -----------------------------------------------------

    nodes, edges = (
        network_to_geodataframes(
            graph
        )
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if not isinstance(
        nodes,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "Network nodes must be a GeoDataFrame."
        )

    if not isinstance(
        edges,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "Network edges must be a GeoDataFrame."
        )

    if nodes.empty:
        raise ValueError(
            f"No road nodes found for {store.city}."
        )

    if edges.empty:
        raise ValueError(
            f"No road edges found for {store.city}."
        )

    return graph, nodes, edges

def build_store_node_intelligence(
    graph,
    nodes,
    store,
    evidence
):
    """
    Build the complete fresh generic node-intelligence
    layer for one store.

    Pipeline:
        graph
        → network connectivity
        → spatial enrichment
        → base node opportunity
        → nearby analytical evidence
        → composite node opportunity

    No city-specific logic.
    """

    if graph is None:
        raise ValueError(
            "graph cannot be None."
        )

    if nodes is None or nodes.empty:
        raise ValueError(
            f"No road nodes available for {store.city}."
        )

    if evidence is None or evidence.empty:
        raise ValueError(
            f"No analytical evidence available for "
            f"{store.city}."
        )

    # -----------------------------------------------------
    # 1. Structural network intelligence
    # -----------------------------------------------------

    network_nodes = build_node_intelligence(
        graph,
        nodes
    )

    if not isinstance(
        network_nodes,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "build_node_intelligence() must return "
            "a GeoDataFrame."
        )

    # -----------------------------------------------------
    # 2. Generic spatial enrichment
    #
    # Creates:
    #     distance_m
    #     catchment_zone
    #     distance_weight
    # -----------------------------------------------------

    spatial_nodes = enrich_network_nodes(
        network_nodes,
        store
    )

    if not isinstance(
        spatial_nodes,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "enrich_network_nodes() must return "
            "a GeoDataFrame."
        )

    # -----------------------------------------------------
    # 3. Base node opportunity
    # -----------------------------------------------------

    opportunity_nodes = calculate_node_opportunity(
        spatial_nodes
    )

    if not isinstance(
        opportunity_nodes,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "calculate_node_opportunity() must return "
            "a GeoDataFrame."
        )

    # -----------------------------------------------------
    # 4. Nearby analytical evidence
    # -----------------------------------------------------

    contextual_nodes = (
        calculate_node_evidence_proximity(
            opportunity_nodes,
            evidence,
            store
        )
    )

    if not isinstance(
        contextual_nodes,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "calculate_node_evidence_proximity() must "
            "return a GeoDataFrame."
        )

    # -----------------------------------------------------
    # 5. Composite node opportunity
    # -----------------------------------------------------

    final_nodes = (
        calculate_composite_node_opportunity(
            contextual_nodes
        )
    )

    if not isinstance(
        final_nodes,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "calculate_composite_node_opportunity() "
            "must return a GeoDataFrame."
        )

    return final_nodes

def build_edge_intelligence(edges):
    """
    Build generic road-segment indicators.

    The current V1 score combines:
        road class
        segment length

    It does NOT claim actual traffic volume.
    """

    result = add_road_class_weight(
        edges
    )

    if "length" in result.columns:

        result["length_km"] = (
            result["length"] / 1000
        )

    else:

        result["length_km"] = 0.0

    result["structural_road_signal"] = (
        result["road_class_weight"]
        * result["length_km"]
    )

    return result

def calculate_segment_evidence_proximity(
    edges,
    evidence,
    store,
    radius_m=150
):
    """
    Attach nearby analytical evidence to each road segment.

    Preserves the original OSMnx edge endpoint information
    (u, v, key) so downstream corridor construction can
    reconstruct connectivity.

    Spatial operations use a local metre-based CRS.
    """

    if edges.empty:
        return edges.copy()

    if radius_m <= 0:
        raise ValueError(
            "radius_m must be positive."
        )

    result = edges.copy()

    # -----------------------------------------------------
    # Preserve original OSMnx edge index
    # -----------------------------------------------------

    original_index = result.index.copy()

    # -----------------------------------------------------
    # Recover u / v from the OSMnx MultiIndex
    # -----------------------------------------------------

    index_names = list(
        result.index.names
    )

    if "u" in result.columns:
        result["_u"] = result["u"]

    elif "u" in index_names:
        result["_u"] = (
            result.index.get_level_values("u")
        )

    else:
        raise ValueError(
            "Input edge data does not contain endpoint 'u'."
        )

    if "v" in result.columns:
        result["_v"] = result["v"]

    elif "v" in index_names:
        result["_v"] = (
            result.index.get_level_values("v")
        )

    else:
        raise ValueError(
            "Input edge data does not contain endpoint 'v'."
        )

    # -----------------------------------------------------
    # Stable row identifier
    # -----------------------------------------------------

    result["_segment_id"] = np.arange(
        len(result)
    )

    # -----------------------------------------------------
    # Initialise output columns
    # -----------------------------------------------------

    result[
        "segment_nearby_feature_count"
    ] = 0

    result[
        "segment_commercial_signal"
    ] = 0.0

    result[
        "segment_movement_signal"
    ] = 0.0

    result[
        "segment_trip_generator_signal"
    ] = 0.0

    # -----------------------------------------------------
    # No evidence available
    # -----------------------------------------------------

    if evidence.empty:

        result.index = original_index

        return result

    # -----------------------------------------------------
    # Local metric CRS
    # -----------------------------------------------------

    local_crs = create_local_projection(
        store
    )

    edges_local = result.to_crs(
        local_crs
    )

    evidence_local = evidence.to_crs(
        local_crs
    )

    # -----------------------------------------------------
    # Buffer each road segment
    # -----------------------------------------------------

    buffered_edges = edges_local[
        [
            "_segment_id",
            "geometry"
        ]
    ].copy()

    buffered_edges["geometry"] = (
        buffered_edges.geometry
        .buffer(radius_m)
    )

    # -----------------------------------------------------
    # Spatial join
    # -----------------------------------------------------

    joined = gpd.sjoin(
        evidence_local,
        buffered_edges,
        predicate="within",
        how="inner"
    )

    if not joined.empty:

        # -------------------------------------------------
        # Aggregate evidence by road segment
        # -------------------------------------------------

        grouped = (
            joined
            .groupby("_segment_id")
            .agg(
                segment_nearby_feature_count=(
                    "feature_category",
                    "count"
                ),
                segment_commercial_signal=(
                    "commercial_spatial_signal",
                    "sum"
                ),
                segment_movement_signal=(
                    "movement_spatial_signal",
                    "sum"
                ),
                segment_trip_generator_signal=(
                    "trip_generator_spatial_signal",
                    "sum"
                )
            )
            .reset_index()
        )

        # -------------------------------------------------
        # Merge on stable row ID
        # -------------------------------------------------

        result = result.merge(
            grouped,
            on="_segment_id",
            how="left",
            suffixes=(
                "",
                "_calculated"
            )
        )

        # -------------------------------------------------
        # Replace initial values with calculated values
        # -------------------------------------------------

        calculated_columns = [
            "segment_nearby_feature_count",
            "segment_commercial_signal",
            "segment_movement_signal",
            "segment_trip_generator_signal"
        ]

        for column in calculated_columns:

            calculated_column = (
                f"{column}_calculated"
            )

            if calculated_column in result.columns:

                result[column] = (
                    result[
                        calculated_column
                    ]
                    .fillna(
                        result[column]
                    )
                )

                result.drop(
                    columns=[
                        calculated_column
                    ],
                    inplace=True
                )

    # -----------------------------------------------------
    # Clean helper column
    # -----------------------------------------------------

    result.drop(
        columns=[
            "_segment_id"
        ],
        inplace=True
    )

    result[
        "segment_nearby_feature_count"
    ] = (
        result[
            "segment_nearby_feature_count"
        ]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # Restore original OSMnx edge index
    # -----------------------------------------------------

    if len(result) != len(original_index):

        raise RuntimeError(
            "Segment row count changed unexpectedly "
            "during spatial enrichment."
        )

    result.index = original_index

    return result

def calculate_segment_opportunity(
    edges
):
    """
    Calculate a relative segment opportunity index.

    Combines:
        road structure
        commercial adjacency
        movement adjacency
        trip-generator adjacency

    The result is a planning signal,
    NOT a traffic-volume estimate.
    """

    if edges.empty:
        return edges.copy()

    result = edges.copy()

    result[
        "road_structure_index"
    ] = min_max_normalize(
        result[
            "structural_road_signal"
        ].fillna(0)
    )

    result[
        "segment_commercial_index"
    ] = min_max_normalize(
        result[
            "segment_commercial_signal"
        ].fillna(0)
    )

    result[
        "segment_movement_index"
    ] = min_max_normalize(
        result[
            "segment_movement_signal"
        ].fillna(0)
    )

    result[
        "segment_trip_generator_index"
    ] = min_max_normalize(
        result[
            "segment_trip_generator_signal"
        ].fillna(0)
    )

    result[
        "segment_opportunity_index"
    ] = (

        result[
            "road_structure_index"
        ]
        * SEGMENT_OPPORTUNITY_WEIGHTS[
            "road_structure"
        ]

        +

        result[
            "segment_commercial_index"
        ]
        * SEGMENT_OPPORTUNITY_WEIGHTS[
            "commercial"
        ]

        +

        result[
            "segment_movement_index"
        ]
        * SEGMENT_OPPORTUNITY_WEIGHTS[
            "movement"
        ]

        +

        result[
            "segment_trip_generator_index"
        ]
        * SEGMENT_OPPORTUNITY_WEIGHTS[
            "trip_generators"
        ]
    )

    return result

def calculate_auto_top_opportunity(
    segments,
    config=MEDIA_OPPORTUNITY_CONFIG
):
    """
    Calculate relative Auto Top opportunity for
    road segments.

    Auto Tops are treated as movement-oriented media.

    This is NOT a traffic-volume estimate.
    """

    if segments.empty:
        return segments.copy()

    result = segments.copy()

    result["auto_movement_index"] = (
        result["segment_movement_index"]
    )

    result["auto_commercial_index"] = (
        result["segment_commercial_index"]
    )

    result["auto_trip_generator_index"] = (
        result[
            "segment_trip_generator_index"
        ]
    )

    result["auto_road_structure_index"] = (
        result["road_structure_index"]
    )

    weights = config["auto_tops"]

    result["auto_top_opportunity"] = (
        result[
            "auto_movement_index"
        ]
        * weights["segment_movement"]

        +

        result[
            "auto_commercial_index"
        ]
        * weights["segment_commercial"]

        +

        result[
            "auto_trip_generator_index"
        ]
        * weights["segment_trip_generators"]

        +

        result[
            "auto_road_structure_index"
        ]
        * weights["road_structure"]
    )

    result["auto_top_index"] = (
        min_max_normalize(
            result[
                "auto_top_opportunity"
            ]
        )
    )

    return result

def build_store_segment_intelligence(
    edges,
    store,
    evidence
):
    """
    Build the complete fresh generic road-segment
    intelligence layer.

    Pipeline:
        raw network edges
        → structural road intelligence
        → nearby analytical evidence
        → segment opportunity

    No city-specific logic.
    """

    if edges is None or edges.empty:
        raise ValueError(
            f"No road edges available for {store.city}."
        )

    if evidence is None or evidence.empty:
        raise ValueError(
            f"No analytical evidence available for "
            f"{store.city}."
        )

    # -----------------------------------------------------
    # 1. Structural road intelligence
    # -----------------------------------------------------

    structural_edges = (
        build_edge_intelligence(
            edges
        )
    )

    if not isinstance(
        structural_edges,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "build_edge_intelligence() must return "
            "a GeoDataFrame."
        )

    # -----------------------------------------------------
    # 2. Nearby analytical evidence
    # -----------------------------------------------------

    contextual_edges = (
        calculate_segment_evidence_proximity(
            structural_edges,
            evidence,
            store
        )
    )

    if not isinstance(
        contextual_edges,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "calculate_segment_evidence_proximity() "
            "must return a GeoDataFrame."
        )

    # -----------------------------------------------------
    # 3. Segment opportunity
    # -----------------------------------------------------

    final_edges = (
        calculate_segment_opportunity(
            contextual_edges
        )
    )

    if not isinstance(
        final_edges,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "calculate_segment_opportunity() must return "
            "a GeoDataFrame."
        )

    return final_edges

def validate_network_output(
    graph,
    nodes,
    edges,
    metrics
):

    if graph is None:
        raise AssertionError(
            "Graph is missing."
        )

    if nodes.empty:
        raise AssertionError(
            "No network nodes returned."
        )

    if edges.empty:
        raise AssertionError(
            "No network edges returned."
        )

    required_metrics = {
        "node_count",
        "edge_count",
        "total_road_length_km",
        "mean_node_degree",
        "median_node_degree",
        "high_connectivity_nodes",
    }

    missing = (
        required_metrics
        - set(metrics.keys())
    )

    if missing:
        raise AssertionError(
            f"Missing network metrics: {sorted(missing)}"
        )

    if metrics["node_count"] <= 0:
        raise AssertionError(
            "Node count must be positive."
        )

    if metrics["edge_count"] <= 0:
        raise AssertionError(
            "Edge count must be positive."
        )

    return True

def validate_node_opportunity(nodes):

    required = {
        "geometry",
        "degree",
        "distance_m",
        "catchment_zone",
        "distance_weight",
        "connectivity_index",
        "node_opportunity",
    }

    missing = required - set(nodes.columns)

    if missing:
        raise AssertionError(
            f"Missing node columns: {sorted(missing)}"
        )

    numeric_columns = [
        "degree",
        "distance_m",
        "distance_weight",
        "connectivity_index",
        "node_opportunity",
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            nodes[column]
        ):
            raise AssertionError(
                f"{column} must be numeric."
            )

        if nodes[column].dropna().lt(0).any():
            raise AssertionError(
                f"Negative values in {column}."
            )

    return True

def build_corridor_candidates(
    edges,
    nodes
):
    """
    Create generic road-segment corridor candidates.

    Works with OSMnx edge GeoDataFrames where:
        u / v are either index levels or columns.

    No city-specific logic.
    """

    if edges.empty:
        return edges.copy()

    result = edges.copy()

    # -----------------------------------------------------
    # Make edge endpoint IDs explicit columns
    # -----------------------------------------------------

    index_names = list(
        result.index.names
    )

    if "u" in result.columns:
        u_values = result["u"]

    elif "u" in index_names:
        u_values = result.index.get_level_values("u")

    else:
        raise ValueError(
            "OSM edge data does not contain endpoint 'u'."
        )

    if "v" in result.columns:
        v_values = result["v"]

    elif "v" in index_names:
        v_values = result.index.get_level_values("v")

    else:
        raise ValueError(
            "OSM edge data does not contain endpoint 'v'."
        )

    result["_u"] = u_values
    result["_v"] = v_values

    # -----------------------------------------------------
    # Build node lookup
    # -----------------------------------------------------

    node_lookup = nodes[
        [
            "connectivity_index",
            "composite_node_opportunity"
        ]
    ].copy()

    node_lookup = node_lookup.to_dict(
        orient="index"
    )

    # -----------------------------------------------------
    # Endpoint structural signals
    # -----------------------------------------------------

    result["u_connectivity"] = (
        result["_u"]
        .map(
            lambda x:
            node_lookup.get(
                x,
                {}
            ).get(
                "connectivity_index",
                0.0
            )
        )
    )

    result["v_connectivity"] = (
        result["_v"]
        .map(
            lambda x:
            node_lookup.get(
                x,
                {}
            ).get(
                "connectivity_index",
                0.0
            )
        )
    )

    result["u_opportunity"] = (
        result["_u"]
        .map(
            lambda x:
            node_lookup.get(
                x,
                {}
            ).get(
                "composite_node_opportunity",
                0.0
            )
        )
    )

    result["v_opportunity"] = (
        result["_v"]
        .map(
            lambda x:
            node_lookup.get(
                x,
                {}
            ).get(
                "composite_node_opportunity",
                0.0
            )
        )
    )

    # -----------------------------------------------------
    # Endpoint averages
    # -----------------------------------------------------

    result["endpoint_connectivity"] = (
        result[
            [
                "u_connectivity",
                "v_connectivity"
            ]
        ]
        .mean(axis=1)
    )

    result["endpoint_opportunity"] = (
        result[
            [
                "u_opportunity",
                "v_opportunity"
            ]
        ]
        .mean(axis=1)
    )

    # -----------------------------------------------------
    # Ensure structural edge signal exists
    # -----------------------------------------------------

    if (
        "structural_road_signal"
        not in result.columns
    ):

        result = build_edge_intelligence(
            result
        )

    # -----------------------------------------------------
    # Generic corridor candidate signal
    # -----------------------------------------------------

    result["corridor_candidate_signal"] = (
        result[
            "structural_road_signal"
        ]
        *
        (
            1
            +
            result[
                "endpoint_opportunity"
            ].fillna(0)
            / 100
        )
    )

    return result

def build_corridor_groups(
    edges,
    graph
):
    """
    Group connected high-opportunity road segments
    into candidate corridors.

    Each connected component becomes a provisional
    corridor group.

    This is a structural grouping step, not yet a
    final named-road interpretation.
    """

    candidates = edges[
        edges["is_corridor_candidate"]
    ].copy()

    if candidates.empty:
        return candidates.copy()

    corridor_graph = nx.Graph()

    # -----------------------------------------------------
    # Add candidate road segments as graph edges
    # -----------------------------------------------------

    for index, row in candidates.iterrows():

        u = row["_u"]
        v = row["_v"]

        corridor_graph.add_edge(
            u,
            v,
            edge_index=index
        )

    # -----------------------------------------------------
    # Connected components
    # -----------------------------------------------------

    components = list(
        nx.connected_components(
            corridor_graph
        )
    )

    node_to_corridor = {}

    for corridor_id, component in enumerate(
        components,
        start=1
    ):

        for node in component:
            node_to_corridor[node] = corridor_id

    # -----------------------------------------------------
    # Assign each candidate segment to a corridor
    # -----------------------------------------------------

    candidates["corridor_id"] = (
        candidates["_u"]
        .map(node_to_corridor)
    )

    return candidates

def aggregate_corridors(edges):
    """
    Convert candidate road segments into provisional
    corridor-level records.

    Each corridor receives:
        segment count
        total length
        mean road-class weight
        mean opportunity
        maximum opportunity
    """

    if edges.empty:
        return pd.DataFrame()

    grouped = (
        edges
        .groupby("corridor_id")
        .agg(
            segment_count=(
                "corridor_id",
                "size"
            ),
            total_length_km=(
                "length_km",
                "sum"
            ),
            mean_road_class_weight=(
                "road_class_weight",
                "mean"
            ),
            mean_opportunity=(
                "corridor_opportunity_index",
                "mean"
            ),
            max_opportunity=(
                "corridor_opportunity_index",
                "max"
            ),
        )
        .reset_index()
    )

    grouped["corridor_score"] = (
        grouped[
            "mean_opportunity"
        ]
        * 0.70
        +
        grouped[
            "max_opportunity"
        ]
        * 0.30
    )

    grouped = grouped.sort_values(
        "corridor_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    grouped["corridor_rank"] = (
        grouped.index + 1
    )

    return grouped

def validate_corridor_groups(
    segments,
    summary
):
    """
    Validate provisional corridor grouping.
    """

    if segments.empty:
        return True

    required_segment_columns = {
        "_u",
        "_v",
        "corridor_id",
        "is_corridor_candidate",
        "geometry"
    }

    missing_segments = (
        required_segment_columns
        - set(segments.columns)
    )

    if missing_segments:
        raise AssertionError(
            "Missing segment columns: "
            f"{sorted(missing_segments)}"
        )

    if summary.empty:
        raise AssertionError(
            "Corridor summary is empty despite "
            "candidate segments existing."
        )

    required_summary_columns = {
        "corridor_id",
        "segment_count",
        "total_length_km",
        "corridor_score",
        "corridor_rank"
    }

    missing_summary = (
        required_summary_columns
        - set(summary.columns)
    )

    if missing_summary:
        raise AssertionError(
            "Missing corridor summary columns: "
            f"{sorted(missing_summary)}"
        )

    if (
        summary["segment_count"]
        .le(0)
        .any()
    ):
        raise AssertionError(
            "Invalid corridor segment count."
        )

    if (
        summary["total_length_km"]
        .lt(0)
        .any()
    ):
        raise AssertionError(
            "Negative corridor length."
        )

    return True

def build_operational_corridors(
    segments
):
    """
    Group connected high-opportunity road segments
    into provisional operational corridors.

    The function explicitly uses endpoint IDs stored in
    _u and _v.

    No city-specific logic.
    """

    if segments.empty:
        return segments.copy()

    result = segments.copy()

    # -----------------------------------------------------
    # Make sure endpoint information exists
    # -----------------------------------------------------

    if "_u" not in result.columns:

        index_names = list(
            result.index.names
        )

        if "u" in index_names:
            result["_u"] = (
                result.index
                .get_level_values("u")
            )

        elif "u" in result.columns:
            result["_u"] = result["u"]

        else:
            raise ValueError(
                "Could not identify road endpoint 'u'."
            )

    if "_v" not in result.columns:

        index_names = list(
            result.index.names
        )

        if "v" in index_names:
            result["_v"] = (
                result.index
                .get_level_values("v")
            )

        elif "v" in result.columns:
            result["_v"] = result["v"]

        else:
            raise ValueError(
                "Could not identify road endpoint 'v'."
            )

    # -----------------------------------------------------
    # Build connectivity graph
    # -----------------------------------------------------

    corridor_graph = nx.Graph()

    for u, v in zip(
        result["_u"],
        result["_v"]
    ):

        corridor_graph.add_edge(
            u,
            v
        )

    # -----------------------------------------------------
    # Find connected components
    # -----------------------------------------------------

    components = list(
        nx.connected_components(
            corridor_graph
        )
    )

    node_to_corridor = {}

    for corridor_id, component in enumerate(
        components,
        start=1
    ):

        for node in component:

            node_to_corridor[node] = (
                corridor_id
            )

    # -----------------------------------------------------
    # Assign corridor ID to each segment
    # -----------------------------------------------------

    result[
        "operational_corridor_id"
    ] = [
        node_to_corridor.get(u)
        for u in result["_u"]
    ]

    return result

def select_operational_corridor_segments(
    segments,
    percentile=0.75
):
    """
    Select higher-opportunity road segments as candidates
    for operational corridor formation.

    The threshold is calculated relative to the supplied
    network, making the rule generic across locations.

    percentile=0.75 means the top 25% of segments by
    Auto Top opportunity are considered.
    """

    if segments.empty:
        return segments.copy()

    if not 0.0 < percentile < 1.0:
        raise ValueError(
            "percentile must be between 0 and 1."
        )

    if "auto_top_index" not in segments.columns:
        raise AssertionError(
            "Missing auto_top_index."
        )

    threshold = (
        segments[
            "auto_top_index"
        ]
        .quantile(percentile)
    )

    candidates = segments[
        segments[
            "auto_top_index"
        ]
        >= threshold
    ].copy()

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if candidates.empty:
        raise ValueError(
            "No operational corridor candidates "
            "were selected."
        )

    candidates[
        "corridor_candidate_threshold"
    ] = threshold

    candidates[
        "is_corridor_candidate"
    ] = True

    return candidates

def summarize_operational_corridors(
    operational_segments
):
    """
    Build a generic operational-corridor summary.
    """

    summary = (
        operational_segments
        .groupby(
            "operational_corridor_id"
        )
        .agg(
            segment_count=(
                "length",
                "count"
            ),
            total_length_km=(
                "length",
                lambda x:
                x.sum() / 1000
            ),
            mean_auto_top_index=(
                "auto_top_index",
                "mean"
            ),
            max_auto_top_index=(
                "auto_top_index",
                "max"
            )
        )
        .reset_index()
    )

    summary[
        "corridor_score"
    ] = (
        summary[
            "mean_auto_top_index"
        ] * 0.70
        +
        summary[
            "max_auto_top_index"
        ] * 0.30
    )

    summary[
        "corridor_rank"
    ] = (
        summary[
            "corridor_score"
        ]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)
    )

    summary = (
        summary
        .sort_values(
            "corridor_rank"
        )
        .reset_index(
            drop=True
        )
    )

    return summary

def build_store_operational_corridors(
    edges,
    nodes,
    store,
    evidence,
    percentile=0.75
):
    """
    Build operational Auto Top corridors from fresh
    road segments.

    Pipeline:
        fresh roads
        → segment evidence
        → segment opportunity
        → Auto Top opportunity
        → top candidate segments
        → corridor grouping

    No city-specific logic.
    """

    # -----------------------------------------------------
    # Auto Top segment opportunity
    # -----------------------------------------------------

    scored_segments = (
        build_store_auto_segment_opportunity(
            edges,
            store,
            evidence
        )
    )

    # -----------------------------------------------------
    # Candidate selection
    # -----------------------------------------------------

    candidates = (
        select_operational_corridor_segments(
            scored_segments,
            percentile=percentile
        )
    )

    # -----------------------------------------------------
    # Corridor candidate enrichment
    # -----------------------------------------------------

    corridor_candidates = (
        build_corridor_candidates(
            candidates,
            nodes
        )
    )

    # -----------------------------------------------------
    # Operational grouping
    # -----------------------------------------------------

    operational = (
        build_operational_corridors(
            corridor_candidates
        )
    )

    return operational

def calculate_corridor_media_opportunity(
    operational_segments,
    operational_corridors
):
    """
    Aggregate segment-level Auto Top opportunity into
    operational corridor-level opportunity.

    Generic and location-independent.

    Handles existing columns in the corridor summary
    without relying on Pandas suffix behaviour.
    """

    if operational_corridors.empty:
        return operational_corridors.copy()

    if operational_segments.empty:
        result = operational_corridors.copy()

        result["corridor_auto_opportunity"] = 0.0
        result["corridor_auto_index"] = 0.0

        return result

    segments = operational_segments.copy()
    result = operational_corridors.copy()

    # -----------------------------------------------------
    # Ensure required segment opportunity fields exist
    # -----------------------------------------------------

    if "auto_top_index" not in segments.columns:

        segments = calculate_auto_top_opportunity(
            segments
        )

    required_segment_columns = {
        "operational_corridor_id",
        "auto_top_index",
        "segment_movement_signal",
        "segment_commercial_signal",
        "segment_trip_generator_signal",
    }

    missing = (
        required_segment_columns
        - set(segments.columns)
    )

    if missing:
        raise AssertionError(
            "Missing required segment fields: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Aggregate ONLY segment data
    # -----------------------------------------------------

    segment_scores = (
        segments
        .groupby(
            "operational_corridor_id"
        )
        .agg(
            corridor_mean_auto_opportunity=(
                "auto_top_index",
                "mean"
            ),
            corridor_max_auto_opportunity=(
                "auto_top_index",
                "max"
            ),
            corridor_movement_signal=(
                "segment_movement_signal",
                "sum"
            ),
            corridor_commercial_signal=(
                "segment_commercial_signal",
                "sum"
            ),
            corridor_trip_generator_signal=(
                "segment_trip_generator_signal",
                "sum"
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Remove old calculated fields from corridor summary
    # if they already exist.
    #
    # This prevents merge collisions.
    # -----------------------------------------------------

    columns_to_replace = [
        "corridor_mean_auto_opportunity",
        "corridor_max_auto_opportunity",
        "corridor_movement_signal",
        "corridor_commercial_signal",
        "corridor_trip_generator_signal",
        "corridor_auto_opportunity",
        "corridor_auto_index",
    ]

    for column in columns_to_replace:

        if column in result.columns:

            result = result.drop(
                columns=[column]
            )

    # -----------------------------------------------------
    # Merge fresh segment-derived evidence
    # -----------------------------------------------------

    result = result.merge(
        segment_scores,
        on="operational_corridor_id",
        how="left"
    )

    # -----------------------------------------------------
    # Fill missing values
    # -----------------------------------------------------

    evidence_columns = [
        "corridor_mean_auto_opportunity",
        "corridor_max_auto_opportunity",
        "corridor_movement_signal",
        "corridor_commercial_signal",
        "corridor_trip_generator_signal",
    ]

    result[evidence_columns] = (
        result[evidence_columns]
        .fillna(0.0)
    )

    # -----------------------------------------------------
    # Normalize supporting evidence
    # -----------------------------------------------------

    movement_index = min_max_normalize(
        result[
            "corridor_movement_signal"
        ]
    )

    commercial_index = min_max_normalize(
        result[
            "corridor_commercial_signal"
        ]
    )

    # -----------------------------------------------------
    # Final corridor Auto opportunity
    # -----------------------------------------------------

    result[
        "corridor_auto_opportunity"
    ] = (

        result[
            "corridor_mean_auto_opportunity"
        ] * 0.50

        +

        result[
            "corridor_max_auto_opportunity"
        ] * 0.20

        +

        movement_index * 0.20

        +

        commercial_index * 0.10
    )

    # -----------------------------------------------------
    # Convert to relative 0–100 index
    # -----------------------------------------------------

    result[
        "corridor_auto_index"
    ] = min_max_normalize(
        result[
            "corridor_auto_opportunity"
        ]
    )

    # -----------------------------------------------------
    # Rank
    # -----------------------------------------------------

    result = (
        result
        .sort_values(
            "corridor_auto_index",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return result

def build_store_corridor_opportunity(
    operational_segments,
    store
):
    """
    Build final corridor-level Auto Top opportunity.

    Uses the existing generic corridor opportunity engine.
    """

    if (
        operational_segments is None
        or operational_segments.empty
    ):
        raise ValueError(
            f"No operational corridor segments "
            f"available for {store.city}."
        )

    # -----------------------------------------------------
    # Create corridor summary from the actual segments
    # -----------------------------------------------------

    corridor_summary = (
        summarize_operational_corridors(
            operational_segments
        )
    )

    # -----------------------------------------------------
    # Existing generic corridor opportunity engine
    #
    # It expects:
    #   operational_segments
    #   operational_corridors
    # -----------------------------------------------------

    corridor_opportunity = (
        calculate_corridor_media_opportunity(
            operational_segments,
            corridor_summary
        )
    )

    if not isinstance(
        corridor_opportunity,
        pd.DataFrame
    ):
        raise TypeError(
            "Corridor opportunity output must be "
            "a pandas DataFrame."
        )

    required = {
        "operational_corridor_id",
        "corridor_auto_opportunity",
        "corridor_auto_index"
    }

    missing = (
        required
        - set(corridor_opportunity.columns)
    )

    if missing:
        raise AssertionError(
            "Missing corridor opportunity columns: "
            f"{sorted(missing)}"
        )

    return corridor_opportunity

def build_store_auto_segment_opportunity(
    edges,
    store,
    evidence
):
    """
    Build the complete generic Auto Top opportunity
    for fresh road segments.

    Pipeline:
        road structure
        → local evidence
        → segment opportunity
        → Auto Top opportunity

    No city-specific logic.
    """

    if edges is None or edges.empty:
        raise ValueError(
            f"No road segments available for {store.city}."
        )

    if evidence is None or evidence.empty:
        raise ValueError(
            f"No analytical evidence available for "
            f"{store.city}."
        )

    # -----------------------------------------------------
    # 1. Structural road intelligence
    # -----------------------------------------------------

    structural_edges = (
        build_edge_intelligence(
            edges
        )
    )

    # -----------------------------------------------------
    # 2. Segment-level evidence proximity
    # -----------------------------------------------------

    contextual_edges = (
        calculate_segment_evidence_proximity(
            structural_edges,
            evidence,
            store
        )
    )

    # -----------------------------------------------------
    # 3. Segment opportunity
    #
    # Creates:
    #   road_structure_index
    #   segment_commercial_index
    #   segment_movement_index
    #   segment_trip_generator_index
    #   segment_opportunity_index
    # -----------------------------------------------------

    opportunity_edges = (
        calculate_segment_opportunity(
            contextual_edges
        )
    )

    # -----------------------------------------------------
    # 4. Auto Top opportunity
    # -----------------------------------------------------

    auto_edges = (
        calculate_auto_top_opportunity(
            opportunity_edges
        )
    )

    if not isinstance(
        auto_edges,
        gpd.GeoDataFrame
    ):
        raise TypeError(
            "calculate_auto_top_opportunity() must "
            "return a GeoDataFrame."
        )

    required = {
        "auto_top_index",
        "segment_movement_index",
        "segment_commercial_index",
        "segment_trip_generator_index"
    }

    missing = (
        required
        - set(auto_edges.columns)
    )

    if missing:
        raise AssertionError(
            "Auto Top segment opportunity is missing "
            f"columns: {sorted(missing)}"
        )

    return auto_edges

def validate_operational_corridors(
    segments
):
    """
    Validate operational corridor assignment.
    """

    required = {
        "_u",
        "_v",
        "operational_corridor_id"
    }

    missing = (
        required
        - set(segments.columns)
    )

    if missing:
        raise AssertionError(
            "Missing operational corridor columns: "
            f"{sorted(missing)}"
        )

    if segments.empty:
        raise AssertionError(
            "Operational corridor table is empty."
        )

    if (
        segments[
            "operational_corridor_id"
        ]
        .isna()
        .any()
    ):
        raise AssertionError(
            "Unassigned operational corridor detected."
        )

    if (
        segments[
            "operational_corridor_id"
        ]
        .nunique()
        < 1
    ):
        raise AssertionError(
            "No operational corridors detected."
        )

    return True

def prepare_kiosk_candidates(
    nodes,
    minimum_index=25.0
):
    """
    Prepare road-network nodes eligible for Pole Kiosk
    placement.

    The function ensures that the Pole Kiosk opportunity
    model exists before filtering candidates.

    No city-specific logic.
    """

    if nodes.empty:
        return nodes.copy()

    result = nodes.copy()

    # -----------------------------------------------------
    # Ensure Pole Kiosk opportunity exists
    # -----------------------------------------------------

    if "pole_kiosk_index" not in result.columns:

        result = calculate_pole_kiosk_opportunity(
            result
        )

    # -----------------------------------------------------
    # Validate required output
    # -----------------------------------------------------

    if "pole_kiosk_index" not in result.columns:

        raise AssertionError(
            "Pole Kiosk opportunity model could not "
            "be created."
        )

    # -----------------------------------------------------
    # Candidate threshold
    # -----------------------------------------------------

    result["is_kiosk_candidate"] = (
        result["pole_kiosk_index"]
        >= minimum_index
    )

    return result[
        result["is_kiosk_candidate"]
    ].copy()

def cluster_kiosk_candidates(
    candidates,
    store,
    cluster_radius_m=150
):
    """
    Group kiosk candidates into compact spatial clusters.

    BUG FIX: the original version only checked a new point's
    distance to a cluster's CURRENT centroid before accepting it --
    but the centroid then moves as more points join, so a cluster
    could "chain" across a dense area and end up with a final
    radius well past cluster_radius_m (this is exactly what
    validate_kiosk_compactness caught on a dense real store). Now
    tests the PROSPECTIVE centroid and radius for the whole
    candidate member set before accepting a point, the same
    approach already used for No Parking Board clustering -- a
    cluster's radius can no longer exceed cluster_radius_m by
    construction, not just usually.

    Generic across all locations.
    """

    if candidates.empty:
        return candidates.copy()

    if cluster_radius_m <= 0:
        raise ValueError(
            "cluster_radius_m must be positive."
        )

    result = candidates.copy()

    # -----------------------------------------------------
    # Project to local metre-based CRS
    # -----------------------------------------------------

    local_crs = create_local_projection(
        store
    )

    local = result.to_crs(
        local_crs
    ).copy()

    # -----------------------------------------------------
    # Process highest-opportunity nodes first
    # -----------------------------------------------------

    ordered_ids = (
        local[
            "pole_kiosk_index"
        ]
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    cluster_members = {}
    cluster_centres = {}

    # -----------------------------------------------------
    # Greedy compact clustering (prospective-radius checked)
    # -----------------------------------------------------

    for node_id in ordered_ids:

        point = local.loc[
            node_id,
            "geometry"
        ]

        best_cluster = None
        best_distance = float("inf")

        for cluster_id in cluster_members:

            members = cluster_members[cluster_id]
            current_centre = cluster_centres[cluster_id]

            distance_to_centre = point.distance(current_centre)

            if distance_to_centre > cluster_radius_m:
                continue

            # Test the prospective centroid for the whole member
            # set, not just distance to the current one.
            prospective_ids = members + [node_id]
            prospective_points = local.loc[prospective_ids, "geometry"]
            prospective_centroid = prospective_points.union_all().centroid
            prospective_radius = (
                prospective_points.distance(prospective_centroid).max()
            )

            if prospective_radius <= cluster_radius_m:
                if distance_to_centre < best_distance:
                    best_cluster = cluster_id
                    best_distance = distance_to_centre

        # -------------------------------------------------
        # Create cluster
        # -------------------------------------------------

        if best_cluster is None:

            cluster_id = (
                len(cluster_members) + 1
            )

            cluster_members[cluster_id] = [node_id]
            cluster_centres[cluster_id] = point

        else:

            cluster_id = best_cluster

            cluster_members[cluster_id].append(node_id)

            member_points = local.loc[
                cluster_members[cluster_id],
                "geometry"
            ]

            cluster_centres[cluster_id] = (
                member_points
                .union_all()
                .centroid
            )

    # -----------------------------------------------------
    # Assign cluster IDs
    # -----------------------------------------------------

    node_to_cluster = {}

    for cluster_id, members in (
        cluster_members.items()
    ):

        for node_id in members:

            node_to_cluster[
                node_id
            ] = cluster_id

    result[
        "kiosk_cluster_id"
    ] = [
        node_to_cluster[
            node_id
        ]
        for node_id in result.index
    ]

    return result


def summarize_kiosk_clusters(
    candidates,
    store
):
    """
    Convert kiosk candidate nodes into operational
    kiosk-area records.

    Includes geographic compactness metrics.

    Generic across all locations.
    """

    if candidates.empty:
        return pd.DataFrame()

    result = candidates.copy()

    local_crs = create_local_projection(
        store
    )

    local = result.to_crs(
        local_crs
    ).copy()

    # -----------------------------------------------------
    # Cluster-level geographic metrics
    # -----------------------------------------------------

    cluster_records = []

    for cluster_id, group in local.groupby(
        "kiosk_cluster_id"
    ):

        # Modern replacement for deprecated unary_union
        centroid = (
            group.geometry
            .union_all()
            .centroid
        )

        distances = (
            group.geometry
            .distance(
                centroid
            )
        )

        # BUG FIX: centroid is in the local metre-based projection
        # (needed for correct distance math above) -- it must be
        # reprojected back to WGS84 before being reported as a
        # latitude/longitude, or downstream consumers (e.g. the map)
        # get metre-offsets mislabelled as degrees.
        centroid_wgs84 = (
            gpd.GeoSeries([centroid], crs=local_crs)
            .to_crs("EPSG:4326")
            .iloc[0]
        )

        cluster_records.append(
            {
                "kiosk_cluster_id": cluster_id,
                "cluster_latitude": centroid_wgs84.y,
                "cluster_longitude": centroid_wgs84.x,
                "cluster_radius_m": distances.max(),
                "node_count": len(group),
                "mean_kiosk_index": group[
                    "pole_kiosk_index"
                ].mean(),
                "max_kiosk_index": group[
                    "pole_kiosk_index"
                ].max(),
                "mean_nearby_features": group[
                    "nearby_feature_count"
                ].mean(),
                "mean_commercial_signal": group[
                    "nearby_commercial_signal"
                ].mean(),
                "mean_movement_signal": group[
                    "nearby_movement_signal"
                ].mean(),
                "mean_trip_generator_signal": group[
                    "nearby_trip_generator_signal"
                ].mean(),
            }
        )

    grouped = pd.DataFrame(
        cluster_records
    )

    # -----------------------------------------------------
    # Supporting evidence indices
    # -----------------------------------------------------

    commercial_index = min_max_normalize(
        grouped[
            "mean_commercial_signal"
        ]
    )

    movement_index = min_max_normalize(
        grouped[
            "mean_movement_signal"
        ]
    )

    trip_index = min_max_normalize(
        grouped[
            "mean_trip_generator_signal"
        ]
    )

    # -----------------------------------------------------
    # Cluster opportunity
    # -----------------------------------------------------

    grouped[
        "kiosk_cluster_score"
    ] = (

        grouped[
            "mean_kiosk_index"
        ] * 0.55

        +

        grouped[
            "max_kiosk_index"
        ] * 0.15

        +

        commercial_index * 0.15

        +

        movement_index * 0.10

        +

        trip_index * 0.05
    )

    grouped[
        "kiosk_cluster_index"
    ] = min_max_normalize(
        grouped[
            "kiosk_cluster_score"
        ]
    )

    # -----------------------------------------------------
    # Rank
    # -----------------------------------------------------

    grouped = (
        grouped
        .sort_values(
            "kiosk_cluster_index",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    grouped[
        "kiosk_cluster_rank"
    ] = (
        grouped.index + 1
    )

    return grouped

def validate_kiosk_compactness(
    clusters,
    max_cluster_radius_m=200
):
    """
    Ensure operational kiosk areas remain spatially compact.
    """

    if clusters.empty:
        return True

    if "cluster_radius_m" not in clusters.columns:
        raise AssertionError(
            "Missing cluster_radius_m."
        )

    if (
        clusters["cluster_radius_m"]
        .lt(0)
        .any()
    ):
        raise AssertionError(
            "Negative cluster radius detected."
        )

    oversized = clusters[
        clusters["cluster_radius_m"]
        > max_cluster_radius_m
    ]

    if not oversized.empty:
        raise AssertionError(
            "Kiosk clusters exceed maximum operational "
            f"radius of {max_cluster_radius_m}m."
        )

    return True

def calculate_pole_kiosk_opportunity(
    nodes,
    config=MEDIA_OPPORTUNITY_CONFIG
):
    """
    Calculate relative Pole Kiosk opportunity
    for road-network nodes.

    Pole Kiosks are treated as point/node-oriented media.
    """

    if nodes.empty:
        return nodes.copy()

    result = nodes.copy()

    weights = config["pole_kiosks"]

    result["kiosk_node_connectivity_index"] = (
        result["connectivity_index"]
    )

    result["kiosk_commercial_index"] = (
        result["nearby_commercial_index"]
    )

    result["kiosk_movement_index"] = (
        result["nearby_movement_index"]
    )

    result["kiosk_trip_generator_index"] = (
        result[
            "nearby_trip_generator_index"
        ]
    )

    result["pole_kiosk_opportunity"] = (

        result[
            "kiosk_node_connectivity_index"
        ]
        * weights["node_connectivity"]

        +

        result[
            "kiosk_commercial_index"
        ]
        * weights["node_commercial"]

        +

        result[
            "kiosk_movement_index"
        ]
        * weights["node_movement"]

        +

        result[
            "kiosk_trip_generator_index"
        ]
        * weights["node_trip_generators"]
    )

    result["pole_kiosk_index"] = (
        min_max_normalize(
            result[
                "pole_kiosk_opportunity"
            ]
        )
    )

    return result

def prepare_no_parking_candidates(
    evidence,
    minimum_index=25.0
):
    """
    Prepare local evidence points eligible for
    No Parking Board placement.

    Uses the generic No Parking Board opportunity
    already calculated.

    No city-specific logic.
    """

    if evidence.empty:
        return evidence.copy()

    result = evidence.copy()

    # -----------------------------------------------------
    # Ensure the board opportunity exists
    # -----------------------------------------------------

    if "no_parking_board_index" not in result.columns:

        result = calculate_no_parking_opportunity(
            result
        )

    if "no_parking_board_index" not in result.columns:

        raise AssertionError(
            "No Parking Board opportunity model "
            "could not be created."
        )

    # -----------------------------------------------------
    # Candidate threshold
    # -----------------------------------------------------

    result["is_board_candidate"] = (
        result["no_parking_board_index"]
        >= minimum_index
    )

    return result[
        result["is_board_candidate"]
    ].copy()

def cluster_no_parking_candidates(
    candidates,
    store,
    cluster_radius_m=150
):
    """
    Create compact operational areas for No Parking Boards.

    A candidate can join an existing cluster only if the
    resulting cluster remains within cluster_radius_m of
    its centroid.

    This prevents elongated / chained clusters.

    Generic across all locations.
    """

    if candidates.empty:
        return candidates.copy()

    if cluster_radius_m <= 0:
        raise ValueError(
            "cluster_radius_m must be positive."
        )

    result = candidates.copy()

    # -----------------------------------------------------
    # Local metric CRS
    # -----------------------------------------------------

    local_crs = create_local_projection(
        store
    )

    local = result.to_crs(
        local_crs
    ).copy()

    # -----------------------------------------------------
    # Process strongest candidates first
    # -----------------------------------------------------

    ordered_ids = (
        local[
            "no_parking_board_index"
        ]
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    cluster_members = {}
    cluster_centres = {}

    # -----------------------------------------------------
    # Greedy compact clustering
    # -----------------------------------------------------

    for candidate_id in ordered_ids:

        point = local.loc[
            candidate_id,
            "geometry"
        ]

        best_cluster = None
        best_distance = float("inf")

        # -------------------------------------------------
        # Evaluate existing clusters
        # -------------------------------------------------

        for cluster_id in cluster_members:

            members = cluster_members[
                cluster_id
            ]

            current_centre = cluster_centres[
                cluster_id
            ]

            distance_to_centre = (
                point.distance(
                    current_centre
                )
            )

            if (
                distance_to_centre
                > cluster_radius_m
            ):
                continue

            # -------------------------------------------------
            # Test the prospective centroid.
            # -------------------------------------------------

            prospective_ids = (
                members
                + [candidate_id]
            )

            prospective_points = local.loc[
                prospective_ids,
                "geometry"
            ]

            prospective_centroid = (
                prospective_points
                .union_all()
                .centroid
            )

            prospective_radius = (
                prospective_points
                .distance(
                    prospective_centroid
                )
                .max()
            )

            # -------------------------------------------------
            # Only accept the cluster if it remains compact.
            # -------------------------------------------------

            if (
                prospective_radius
                <= cluster_radius_m
            ):

                if (
                    distance_to_centre
                    < best_distance
                ):

                    best_cluster = cluster_id
                    best_distance = (
                        distance_to_centre
                    )

        # -----------------------------------------------------
        # Create a new cluster if no existing cluster works.
        # -----------------------------------------------------

        if best_cluster is None:

            new_cluster_id = (
                len(cluster_members) + 1
            )

            cluster_members[
                new_cluster_id
            ] = [
                candidate_id
            ]

            cluster_centres[
                new_cluster_id
            ] = point

        else:

            cluster_members[
                best_cluster
            ].append(
                candidate_id
            )

            member_points = local.loc[
                cluster_members[
                    best_cluster
                ],
                "geometry"
            ]

            cluster_centres[
                best_cluster
            ] = (
                member_points
                .union_all()
                .centroid
            )

    # -----------------------------------------------------
    # Map candidate → cluster
    # -----------------------------------------------------

    node_to_cluster = {}

    for cluster_id, members in (
        cluster_members.items()
    ):

        for candidate_id in members:

            node_to_cluster[
                candidate_id
            ] = cluster_id

    # -----------------------------------------------------
    # Store operational cluster ID
    # -----------------------------------------------------

    result[
        "board_cluster_id"
    ] = [
        node_to_cluster[
            candidate_id
        ]
        for candidate_id in result.index
    ]

    return result

def summarize_board_clusters(
    candidates,
    store
):
    """
    Convert board candidate evidence into operational
    No Parking Board areas.

    Uses mean/max opportunity as the primary signal and
    local evidence as supporting information.
    """

    if candidates.empty:
        return pd.DataFrame()

    local_crs = create_local_projection(
        store
    )

    local = candidates.to_crs(
        local_crs
    ).copy()

    cluster_records = []

    for cluster_id, group in local.groupby(
        "board_cluster_id"
    ):

        centroid = (
            group.geometry
            .union_all()
            .centroid
        )

        distances = (
            group.geometry
            .distance(
                centroid
            )
        )

        # BUG FIX: same issue as summarize_kiosk_clusters -- centroid
        # is in the local metre-based projection and must be
        # reprojected back to WGS84 before being reported as a
        # latitude/longitude.
        centroid_wgs84 = (
            gpd.GeoSeries([centroid], crs=local_crs)
            .to_crs("EPSG:4326")
            .iloc[0]
        )

        cluster_records.append(
            {
                "board_cluster_id": cluster_id,

                "cluster_latitude":
                    centroid_wgs84.y,

                "cluster_longitude":
                    centroid_wgs84.x,

                "cluster_radius_m":
                    distances.max(),

                "evidence_count":
                    len(group),

                "mean_board_index":
                    group[
                        "no_parking_board_index"
                    ].mean(),

                "max_board_index":
                    group[
                        "no_parking_board_index"
                    ].max(),

                "mean_commercial_signal":
                    group[
                        "commercial_spatial_signal"
                    ].mean(),

                "mean_movement_signal":
                    group[
                        "movement_spatial_signal"
                    ].mean(),

                "mean_trip_generator_signal":
                    group[
                        "trip_generator_spatial_signal"
                    ].mean(),
            }
        )

    grouped = pd.DataFrame(
        cluster_records
    )

    # -----------------------------------------------------
    # Supporting evidence normalization
    # -----------------------------------------------------

    commercial_index = min_max_normalize(
        grouped[
            "mean_commercial_signal"
        ]
    )

    movement_index = min_max_normalize(
        grouped[
            "mean_movement_signal"
        ]
    )

    trip_index = min_max_normalize(
        grouped[
            "mean_trip_generator_signal"
        ]
    )

    # -----------------------------------------------------
    # Cluster score
    # -----------------------------------------------------

    grouped[
        "board_cluster_score"
    ] = (

        grouped[
            "mean_board_index"
        ] * 0.55

        +

        grouped[
            "max_board_index"
        ] * 0.15

        +

        commercial_index * 0.15

        +

        movement_index * 0.10

        +

        trip_index * 0.05
    )

    # -----------------------------------------------------
    # Relative 0–100 index
    # -----------------------------------------------------

    grouped[
        "board_cluster_index"
    ] = min_max_normalize(
        grouped[
            "board_cluster_score"
        ]
    )

    # -----------------------------------------------------
    # Ranking
    # -----------------------------------------------------

    grouped = (
        grouped
        .sort_values(
            "board_cluster_index",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    grouped[
        "board_cluster_rank"
    ] = (
        grouped.index + 1
    )

    return grouped

def validate_board_compactness(
    clusters,
    max_cluster_radius_m=200
):
    """
    Validate that operational board areas remain
    geographically compact.
    """

    if clusters.empty:
        return True

    required = {
        "board_cluster_id",
        "cluster_radius_m",
        "board_cluster_index"
    }

    missing = (
        required
        - set(clusters.columns)
    )

    if missing:
        raise AssertionError(
            "Missing board cluster columns: "
            f"{sorted(missing)}"
        )

    if (
        clusters["cluster_radius_m"]
        .lt(0)
        .any()
    ):
        raise AssertionError(
            "Negative board cluster radius detected."
        )

    oversized = clusters[
        clusters["cluster_radius_m"]
        > max_cluster_radius_m
    ]

    if not oversized.empty:
        raise AssertionError(
            "Board clusters exceed maximum radius "
            f"of {max_cluster_radius_m}m."
        )

    return True

def build_board_zone_opportunity(
    density,
    store
):
    """
    Build No Parking Board opportunity at catchment-zone
    level.

    This avoids relying on sparse individual evidence
    points and provides a stable operational geography
    for large board packages.

    Generic across all locations.
    """

    if density.empty:
        return pd.DataFrame()

    result = density.copy()

    required_columns = {
        "zone",
        "commercial_density",
        "movement_density",
        "trip_generator_density",
        "area_km2"
    }

    missing = (
        required_columns
        - set(result.columns)
    )

    if missing:
        raise AssertionError(
            "Missing board zone inputs: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Normalize zone signals
    # -----------------------------------------------------

    commercial_index = min_max_normalize(
        result[
            "commercial_density"
        ]
    )

    movement_index = min_max_normalize(
        result[
            "movement_density"
        ]
    )

    trip_index = min_max_normalize(
        result[
            "trip_generator_density"
        ]
    )

    # -----------------------------------------------------
    # Board opportunity
    #
    # Commercial activity is the primary signal.
    # -----------------------------------------------------

    result[
        "board_zone_opportunity"
    ] = (

        commercial_index * 0.55

        +

        movement_index * 0.20

        +

        trip_index * 0.15

        +

        min_max_normalize(
            1 / result["area_km2"]
        ) * 0.10
    )

    # -----------------------------------------------------
    # Convert to 0–100
    # -----------------------------------------------------

    result[
        "board_zone_index"
    ] = min_max_normalize(
        result[
            "board_zone_opportunity"
        ]
    )

    # -----------------------------------------------------
    # Rank
    # -----------------------------------------------------

    result = (
        result
        .sort_values(
            "board_zone_index",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "board_zone_rank"
    ] = (
        result.index + 1
    )

    return result

def calculate_no_parking_opportunity(
    evidence,
    config=MEDIA_OPPORTUNITY_CONFIG
):
    """
    Calculate relative No Parking Board opportunity
    for local areas.

    This model is intentionally based on local
    activity evidence rather than road structure.
    """

    if evidence.empty:
        return evidence.copy()

    result = evidence.copy()

    weights = config["no_parking_boards"]

    result["board_commercial_index"] = (
        min_max_normalize(
            result[
                "commercial_spatial_signal"
            ]
        )
    )

    result["board_movement_index"] = (
        min_max_normalize(
            result[
                "movement_spatial_signal"
            ]
        )
    )

    result["board_trip_generator_index"] = (
        min_max_normalize(
            result[
                "trip_generator_spatial_signal"
            ]
        )
    )

    result["local_feature_density"] = (
        result["catchment_zone"]
        .map(
            result[
                "catchment_zone"
            ]
            .value_counts()
        )
        .fillna(0)
    )

    result["local_feature_density_index"] = (
        min_max_normalize(
            result["local_feature_density"]
        )
    )

    result["no_parking_board_opportunity"] = (

        result[
            "board_commercial_index"
        ]
        * weights["commercial"]

        +

        result[
            "board_movement_index"
        ]
        * weights["movement"]

        +

        result[
            "board_trip_generator_index"
        ]
        * weights["trip_generators"]

        +

        result[
            "local_feature_density_index"
        ]
        * weights["local_feature_density"]
    )

    result["no_parking_board_index"] = (
        min_max_normalize(
            result[
                "no_parking_board_opportunity"
            ]
        )
    )

    return result

def validate_media_opportunity(
    auto_data,
    kiosk_data,
    board_data
):

    required_auto = {
        "auto_top_opportunity",
        "auto_top_index",
        "geometry"
    }

    required_kiosk = {
        "pole_kiosk_opportunity",
        "pole_kiosk_index",
        "geometry"
    }

    required_board = {
        "no_parking_board_opportunity",
        "no_parking_board_index",
        "geometry"
    }

    if not required_auto.issubset(
        auto_data.columns
    ):
        raise AssertionError(
            "Auto Top opportunity output incomplete."
        )

    if not required_kiosk.issubset(
        kiosk_data.columns
    ):
        raise AssertionError(
            "Pole Kiosk opportunity output incomplete."
        )

    if not required_board.issubset(
        board_data.columns
    ):
        raise AssertionError(
            "No Parking Board opportunity output incomplete."
        )

    for data, column in [
        (
            auto_data,
            "auto_top_index"
        ),
        (
            kiosk_data,
            "pole_kiosk_index"
        ),
        (
            board_data,
            "no_parking_board_index"
        )
    ]:

        if data[column].dropna().lt(0).any():
            raise AssertionError(
                f"Negative values in {column}."
            )

    return True

def largest_remainder_allocate(
    scores,
    total_units,
    minimum_units=1,
    maximum_share=1.0
):
    """
    Robust integer allocation of a finite media package.

    Guarantees:
        1. Integer allocations
        2. No negative allocations
        3. Exact reconciliation to total_units
           when configured capacity permits it
        4. Deterministic largest-remainder distribution

    Parameters
    ----------
    scores : pd.Series
        Opportunity score for each candidate.

    total_units : int
        Total inventory available.

    minimum_units : int
        Minimum units for an activated candidate.

    maximum_share : float
        Maximum proportion of total inventory allowed
        for one candidate.

    Returns
    -------
    pd.Series
        Integer allocation indexed like scores.
    """

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    if total_units < 0:
        raise ValueError(
            "total_units cannot be negative."
        )

    if minimum_units < 0:
        raise ValueError(
            "minimum_units cannot be negative."
        )

    if not 0 < maximum_share <= 1:
        raise ValueError(
            "maximum_share must be > 0 and <= 1."
        )

    if scores.empty:

        if total_units == 0:

            return pd.Series(
                dtype=int,
                index=scores.index
            )

        raise ValueError(
            "No eligible candidates available "
            "for positive inventory."
        )

    # -----------------------------------------------------
    # Clean scores
    # -----------------------------------------------------

    scores = pd.to_numeric(
        scores,
        errors="coerce"
    ).fillna(0.0)

    scores = scores.clip(
        lower=0.0
    )

    # -----------------------------------------------------
    # Only positive-opportunity candidates participate
    # -----------------------------------------------------

    positive_scores = scores[
        scores > 0
    ].copy()

    if positive_scores.empty:

        if total_units == 0:

            return pd.Series(
                0,
                index=scores.index,
                dtype=int
            )

        raise ValueError(
            "All candidate opportunity scores are zero."
        )

    # -----------------------------------------------------
    # Calculate maximum units per candidate
    # -----------------------------------------------------

    max_units = int(
        np.floor(
            total_units
            * maximum_share
        )
    )

    # At least one unit must be possible when
    # maximum_share > 0 and inventory > 0.
    max_units = max(
        1,
        max_units
    )

    # -----------------------------------------------------
    # Number of candidates we can activate
    # -----------------------------------------------------

    max_possible_candidates = (
        total_units
        // minimum_units
        if minimum_units > 0
        else len(positive_scores)
    )

    if max_possible_candidates <= 0:

        raise ValueError(
            "Inventory is insufficient for the "
            "configured minimum allocation."
        )

    # -----------------------------------------------------
    # Select candidates.
    #
    # If there isn't enough inventory to activate all
    # candidates, take the highest-scoring ones.
    # -----------------------------------------------------

    if (
        minimum_units > 0
        and
        len(positive_scores)
        * minimum_units
        > total_units
    ):

        selected = (
            positive_scores
            .sort_values(
                ascending=False
            )
            .head(
                max_possible_candidates
            )
            .index
        )

    else:

        selected = positive_scores.index

    selected_scores = (
        positive_scores
        .loc[selected]
        .copy()
    )

    # -----------------------------------------------------
    # Check total capacity.
    # -----------------------------------------------------

    if (
        len(selected_scores)
        * max_units
        < total_units
    ):

        raise ValueError(
            "Allocation is impossible under the "
            "current maximum_target_share. "
            f"Total capacity = "
            f"{len(selected_scores) * max_units}, "
            f"required = {total_units}."
        )

    # -----------------------------------------------------
    # Start with minimum allocation.
    # -----------------------------------------------------

    allocation = pd.Series(
        minimum_units,
        index=selected_scores.index,
        dtype=int
    )

    remaining = (
        total_units
        - allocation.sum()
    )

    if remaining < 0:

        raise ValueError(
            "Minimum allocations exceed available inventory."
        )

    if remaining == 0:

        return allocation.reindex(
            scores.index,
            fill_value=0
        ).astype(int)

    # -----------------------------------------------------
    # Proportional ideal allocation for remaining units
    # -----------------------------------------------------

    proportions = (
        selected_scores
        / selected_scores.sum()
    )

    ideal_extra = (
        proportions
        * remaining
    )

    # -----------------------------------------------------
    # Floor allocation
    # -----------------------------------------------------

    extra = (
        np.floor(
            ideal_extra
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # Respect maximum cap
    # -----------------------------------------------------

    for index in extra.index:

        allowed = (
            max_units
            - allocation.loc[index]
        )

        extra.loc[index] = min(
            extra.loc[index],
            max(
                0,
                allowed
            )
        )

    allocation += extra

    remaining_after_floor = (
        total_units
        - allocation.sum()
    )

    # -----------------------------------------------------
    # Largest remainder pass
    # -----------------------------------------------------

    fractional_remainders = (
        ideal_extra
        - extra
    )

    while remaining_after_floor > 0:

        eligible = [
            index
            for index in
            fractional_remainders
            .sort_values(
                ascending=False
            )
            .index
            if allocation.loc[index] < max_units
        ]

        if not eligible:

            raise ValueError(
                "Unable to reconcile allocation. "
                "No remaining candidate capacity."
            )

        assigned_this_round = False

        for index in eligible:

            if remaining_after_floor <= 0:
                break

            if allocation.loc[index] >= max_units:
                continue

            allocation.loc[index] += 1

            remaining_after_floor -= 1
            assigned_this_round = True

        if not assigned_this_round:

            raise ValueError(
                "Allocation reconciliation stalled."
            )

    # -----------------------------------------------------
    # Final reconciliation
    # -----------------------------------------------------

    if int(allocation.sum()) != int(total_units):

        raise RuntimeError(
            "Final allocation reconciliation failed."
        )

    return allocation.reindex(
        scores.index,
        fill_value=0
    ).astype(int)

def allocate_media(
    dataframe,
    opportunity_column,
    total_units,
    media_type,
    config=None
):
    """
    Generic integer package allocation.

    Logic:
        1. Rank all targets by opportunity.
        2. Prefer targets above the configured threshold.
        3. Expand to lower-ranked targets when required to
           create enough allocation capacity.
        4. Use configured maximum share.
        5. Reconcile exactly to total package.

    Generic across cities and media types.
    """

    if config is None:
        config = ALLOCATION_CONFIG

    if dataframe.empty:
        raise ValueError(
            f"No candidates available for {media_type}."
        )

    if total_units < 0:
        raise ValueError(
            "total_units cannot be negative."
        )

    if opportunity_column not in dataframe.columns:
        raise ValueError(
            f"Missing opportunity column: "
            f"{opportunity_column}"
        )

    result = dataframe.copy()

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    minimum_opportunity = float(
        config[
            "minimum_opportunity_index"
        ][media_type]
    )

    candidate_limit = int(
        config[
            "candidate_limits"
        ][media_type]
    )

    maximum_share = float(
        config[
            "maximum_target_share"
        ][media_type]
    )

    minimum_units = int(
        config[
            "minimum_units"
        ][media_type]
    )

    # -----------------------------------------------------
    # Clean and rank all targets
    # -----------------------------------------------------

    scores = pd.to_numeric(
        result[
            opportunity_column
        ],
        errors="coerce"
    ).fillna(0.0)

    scores = scores.clip(
        lower=0.0
    )

    ranked = (
        pd.DataFrame(
            {
                "score": scores
            },
            index=result.index
        )
        .sort_values(
            "score",
            ascending=False
        )
    )

    # -----------------------------------------------------
    # Maximum allocation per target
    # -----------------------------------------------------

    if total_units == 0:

        result[
            f"{media_type}_allocation"
        ] = 0

        result[
            f"{media_type}_allocation_share"
        ] = 0.0

        return result

    max_units_per_target = max(
        1,
        int(
            np.floor(
                total_units
                * maximum_share
            )
        )
    )

    required_targets = int(
        np.ceil(
            total_units
            / max_units_per_target
        )
    )

    # -----------------------------------------------------
    # Preferred candidate pool
    # -----------------------------------------------------

    preferred = ranked[
        ranked["score"]
        >= minimum_opportunity
    ].head(
        candidate_limit
    )

    # -----------------------------------------------------
    # Expand if preferred candidates lack capacity.
    #
    # Crucially, expansion uses ALL ranked targets,
    # including low-score zones.
    # -----------------------------------------------------

    if (
        len(preferred)
        * max_units_per_target
        >= total_units
    ):

        candidates = preferred.copy()

    else:

        expansion_count = max(
            required_targets,
            len(preferred)
        )

        expansion_count = min(
            expansion_count,
            len(ranked)
        )

        candidates = ranked.head(
            expansion_count
        ).copy()

    # -----------------------------------------------------
    # Final capacity check
    # -----------------------------------------------------

    capacity = (
        len(candidates)
        * max_units_per_target
    )

    if capacity < total_units:

        raise ValueError(
            f"Allocation is impossible for "
            f"{media_type}. "
            f"Targets available = "
            f"{len(candidates)}, "
            f"capacity = {capacity}, "
            f"required = {total_units}. "
            f"Maximum share = "
            f"{maximum_share:.0%}."
        )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Do not give minimum units to zero-score targets
    # unless there is no alternative.
    # -----------------------------------------------------

    allocation_scores = candidates[
        "score"
    ].copy()

    positive_scores = (
        allocation_scores
        .where(
            allocation_scores > 0,
            0.0
        )
    )

    # If every score is zero, distribute evenly.
    if positive_scores.sum() == 0:

        positive_scores[:] = 1.0

    # -----------------------------------------------------
    # Allocate
    # -----------------------------------------------------

    allocations = largest_remainder_allocate(
        scores=positive_scores,
        total_units=total_units,
        minimum_units=minimum_units,
        maximum_share=maximum_share
    )

    # -----------------------------------------------------
    # Attach to original dataframe
    # -----------------------------------------------------

    allocation_column = (
        f"{media_type}_allocation"
    )

    share_column = (
        f"{media_type}_allocation_share"
    )

    result[
        allocation_column
    ] = 0

    result.loc[
        allocations.index,
        allocation_column
    ] = allocations

    result[
        share_column
    ] = (
        result[
            allocation_column
        ]
        / total_units
    )

    return result

def validate_allocation(
    dataframe,
    allocation_column,
    expected_total
):
    """
    Ensure allocated units exactly equal the
    available inventory.
    """

    if allocation_column not in dataframe.columns:
        raise AssertionError(
            f"Missing allocation column: "
            f"{allocation_column}"
        )

    total_allocated = int(
        dataframe[
            allocation_column
        ].sum()
    )

    if total_allocated != expected_total:
        raise AssertionError(
            f"{allocation_column}: "
            f"allocated {total_allocated}, "
            f"expected {expected_total}"
        )

    if dataframe[
        allocation_column
    ].lt(0).any():

        raise AssertionError(
            f"Negative allocation found in "
            f"{allocation_column}"
        )

    return True

def explain_signal_drivers(dataframe, signal_labels):
    """
    CONSOLIDATION ADDITION: short, plain-language reasoning for why a
    candidate (corridor / cluster / zone) scored the way it did --
    which of its supporting signals is strongest RELATIVE TO THE
    OTHER CANDIDATES in this store's own result. Same "relative to
    this location only" philosophy as every other index in this
    engine (see min_max_normalize) -- not an absolute claim about
    the area.

    signal_labels: dict mapping {column_name: human_readable_label}
    for the signal columns to compare, e.g.
        {"corridor_commercial_signal": "commercial activity", ...}
    """

    if dataframe.empty:
        return pd.Series(dtype=str)

    normalized = {
        column: min_max_normalize(dataframe[column].fillna(0.0))
        for column in signal_labels
    }

    explanations = []

    for row_index in dataframe.index:

        scored = sorted(
            (
                (label, normalized[column].loc[row_index])
                for column, label in signal_labels.items()
            ),
            key=lambda pair: pair[1],
            reverse=True
        )

        top_label, top_score = scored[0]
        second_label, second_score = scored[1]

        if top_score < 35:
            explanation = (
                "Weak signal overall -- lowest-confidence "
                "pick in this set"
            )
        elif top_score - second_score < 15 and second_score >= 45:
            explanation = f"Driven by {top_label} and {second_label}"
        else:
            explanation = f"Driven mainly by {top_label}"

        explanations.append(explanation)

    return pd.Series(explanations, index=dataframe.index)


def create_store_input(
    city,
    latitude,
    longitude,
    auto_tops,
    pole_kiosks,
    no_parking_boards
):
    """
    Create and validate a generic store object.
    """

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    if not city or not str(city).strip():

        raise ValueError(
            "city must be provided."
        )

    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:

        raise ValueError(
            "latitude must be between -90 and 90."
        )

    if not -180 <= longitude <= 180:

        raise ValueError(
            "longitude must be between -180 and 180."
        )

    package = {
        "auto_tops": int(auto_tops),
        "pole_kiosks": int(pole_kiosks),
        "no_parking_boards": int(
            no_parking_boards
        )
    }

    for media, quantity in package.items():

        if quantity < 0:

            raise ValueError(
                f"{media} cannot be negative."
            )

    # -----------------------------------------------------
    # Create attribute-based store object
    # -----------------------------------------------------

    return StoreInput(
        city=str(city).strip(),
        latitude=latitude,
        longitude=longitude,
        auto_tops=package["auto_tops"],
        pole_kiosks=package["pole_kiosks"],
        no_parking_boards=package[
            "no_parking_boards"
        ]
    )

def validate_auto_top_allocation(
    dataframe,
    expected_total
):
    """
    Validate integer Auto Top allocation and exact
    package reconciliation.
    """

    allocation_column = (
        "auto_tops_allocation"
    )

    share_column = (
        "auto_tops_allocation_share"
    )

    required = {
        allocation_column,
        share_column,
        "corridor_auto_index"
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise AssertionError(
            "Missing Auto Top allocation columns: "
            f"{sorted(missing)}"
        )

    allocation = dataframe[
        allocation_column
    ]

    if allocation.isna().any():
        raise AssertionError(
            "Missing Auto Top allocation."
        )

    if (allocation < 0).any():
        raise AssertionError(
            "Negative Auto Top allocation detected."
        )

    if (allocation % 1 != 0).any():
        raise AssertionError(
            "Auto Top allocation must be integer."
        )

    allocated_total = int(
        allocation.sum()
    )

    if allocated_total != int(
        expected_total
    ):
        raise AssertionError(
            "Auto Top allocation does not reconcile. "
            f"Allocated = {allocated_total}, "
            f"Expected = {expected_total}"
        )

    share_total = float(
        dataframe[
            share_column
        ].sum()
    )

    if not np.isclose(
        share_total,
        1.0,
        atol=1e-9
    ):
        raise AssertionError(
            "Auto Top allocation shares do not "
            f"reconcile to 1.0. "
            f"Current total = {share_total}"
        )

    return True

def validate_board_zone_allocation(
    dataframe,
    expected_total
):
    """
    Validate the final No Parking Board allocation.

    Checks:
        - allocation column exists
        - allocations are numeric
        - allocations are non-negative
        - allocations are integers
        - package reconciles exactly
        - allocation shares reconcile to 1.0
    """

    allocation_column = (
        "no_parking_boards_allocation"
    )

    share_column = (
        "no_parking_boards_allocation_share"
    )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    if allocation_column not in dataframe.columns:

        raise AssertionError(
            f"Missing allocation column: "
            f"{allocation_column}"
        )

    if share_column not in dataframe.columns:

        raise AssertionError(
            f"Missing allocation share column: "
            f"{share_column}"
        )

    # -----------------------------------------------------
    # Numeric validation
    # -----------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        dataframe[allocation_column]
    ):

        raise AssertionError(
            "Board allocation must be numeric."
        )

    # -----------------------------------------------------
    # Negative allocation check
    # -----------------------------------------------------

    if (
        dataframe[
            allocation_column
        ]
        .lt(0)
        .any()
    ):

        raise AssertionError(
            "Negative board allocation detected."
        )

    # -----------------------------------------------------
    # Integer allocation check
    # -----------------------------------------------------

    if (
        dataframe[
            allocation_column
        ]
        % 1
        != 0
    ).any():

        raise AssertionError(
            "Board allocation contains "
            "non-integer values."
        )

    # -----------------------------------------------------
    # Exact package reconciliation
    # -----------------------------------------------------

    allocated_total = int(
        dataframe[
            allocation_column
        ].sum()
    )

    if allocated_total != int(
        expected_total
    ):

        raise AssertionError(
            "Board package does not reconcile. "
            f"Allocated = {allocated_total}, "
            f"Expected = {expected_total}"
        )

    # -----------------------------------------------------
    # Allocation-share reconciliation
    # -----------------------------------------------------

    share_total = float(
        dataframe[
            share_column
        ].sum()
    )

    if not np.isclose(
        share_total,
        1.0,
        atol=1e-9
    ):

        raise AssertionError(
            "Board allocation shares do not "
            f"reconcile to 1.0. "
            f"Current total = {share_total}"
        )

    # -----------------------------------------------------
    # No NaN allocations
    # -----------------------------------------------------

    if (
        dataframe[
            allocation_column
        ]
        .isna()
        .any()
    ):

        raise AssertionError(
            "Missing board allocation detected."
        )

    return True

def validate_final_store_plan(
    store,
    auto_plan,
    kiosk_plan,
    board_plan
):
    """
    Validate that all three media plans exist for one
    generic store and reconcile to the requested package.
    """

    if store is None:
        raise AssertionError(
            "Store input is missing."
        )

    required_plan_objects = {
        "auto_plan": auto_plan,
        "kiosk_plan": kiosk_plan,
        "board_plan": board_plan
    }

    for name, plan in required_plan_objects.items():

        if plan is None:
            raise AssertionError(
                f"{name} is missing."
            )

        if not isinstance(
            plan,
            pd.DataFrame
        ):
            raise AssertionError(
                f"{name} must be a DataFrame."
            )

    # -----------------------------------------------------
    # Auto Tops
    # -----------------------------------------------------

    validate_auto_top_allocation(
        auto_plan,
        store.auto_tops
    )

    # -----------------------------------------------------
    # Pole Kiosks
    # -----------------------------------------------------

    validate_allocation(
        kiosk_plan,
        "pole_kiosks_allocation",
        store.pole_kiosks
    )

    # -----------------------------------------------------
    # No Parking Boards
    # -----------------------------------------------------

    validate_board_zone_allocation(
        board_plan,
        store.no_parking_boards
    )

    return True

def analyze_store(
    city,
    latitude,
    longitude,
    auto_tops,
    pole_kiosks,
    no_parking_boards,
    radius_m=3000
):
    """
    Production Tom and Jerry pipeline.

    Features are retrieved once.
    Kiosk clustering uses the non-deprecated
    union_all() implementation.
    """

    # -----------------------------------------------------
    # 1. Store input
    # -----------------------------------------------------

    store = create_store_input(
        city=city,
        latitude=latitude,
        longitude=longitude,
        auto_tops=auto_tops,
        pole_kiosks=pole_kiosks,
        no_parking_boards=no_parking_boards
    )

    print(
        f"Starting Tom and Jerry analysis: "
        f"{store.city}"
    )

    # -----------------------------------------------------
    # 2. Single OSM retrieval + preparation
    # -----------------------------------------------------

    raw_osm, prepared_features = (
        build_store_osm_layer(
            store
        )
    )

    print(
        f"Raw OSM features returned: "
        f"{len(raw_osm):,}"
    )

    print(
        f"Prepared features: "
        f"{len(prepared_features):,}"
    )

    # -----------------------------------------------------
    # 3. Feature enrichment
    # -----------------------------------------------------

    enriched_features = enrich_store_features(
        prepared_features,
        store
    )

    # -----------------------------------------------------
    # 4. Analytical evidence
    # -----------------------------------------------------

    analytical_evidence = (
        build_store_analytical_evidence(
            enriched_features,
            store
        )
    )

    # -----------------------------------------------------
    # 5. Catchments
    # -----------------------------------------------------

    catchments = build_store_catchments(
        store
    )

    # CONSOLIDATION FIX: this stage-level validator existed in the
    # original notebook but was never actually called from the
    # production path -- only the final allocation numbers were
    # checked, not the catchment geometry that everything downstream
    # depends on.
    validate_catchments(catchments)

    zoned_features = (
        assign_features_to_catchments(
            prepared_features,
            store,
            catchments
        )
    )

    # -----------------------------------------------------
    # 6. Zone density
    # -----------------------------------------------------

    zone_density = build_store_zone_density(
        analytical_evidence,
        store
    )

    # -----------------------------------------------------
    # 7. Road network
    # -----------------------------------------------------

    graph, nodes, edges = (
        build_store_road_network(
            store,
            radius_m=radius_m
        )
    )

    # -----------------------------------------------------
    # 8. Node intelligence
    # -----------------------------------------------------

    node_intelligence = (
        build_store_node_intelligence(
            graph,
            nodes,
            store,
            analytical_evidence
        )
    )

    # -----------------------------------------------------
    # 9. Segment intelligence
    # -----------------------------------------------------

    segment_intelligence = (
        build_store_segment_intelligence(
            edges,
            store,
            analytical_evidence
        )
    )

    # -----------------------------------------------------
    # 10. Operational corridors
    # -----------------------------------------------------

    operational_segments = (
        build_store_operational_corridors(
            edges,
            node_intelligence,
            store,
            analytical_evidence
        )
    )

    # -----------------------------------------------------
    # 11. Corridor opportunity
    # -----------------------------------------------------

    corridor_opportunity = (
        build_store_corridor_opportunity(
            operational_segments,
            store
        )
    )

    # -----------------------------------------------------
    # 12. Auto Tops
    # -----------------------------------------------------

    auto_plan = allocate_media(
        dataframe=corridor_opportunity,
        opportunity_column="corridor_auto_index",
        total_units=store.auto_tops,
        media_type="auto_tops"
    )

    # CONSOLIDATION FIX: short plain-language reasoning per corridor.
    auto_plan["why"] = explain_signal_drivers(
        auto_plan,
        {
            "corridor_commercial_signal": "commercial activity",
            "corridor_movement_signal": "movement & transit",
            "corridor_trip_generator_signal": "footfall drivers"
        }
    )

    # -----------------------------------------------------
    # 13. Pole Kiosks
    # -----------------------------------------------------

    kiosk_candidates = prepare_kiosk_candidates(
        node_intelligence
    )

    kiosk_candidates = cluster_kiosk_candidates(
        kiosk_candidates,
        store,
        cluster_radius_m=150
    )

    kiosk_clusters = summarize_kiosk_clusters(
        kiosk_candidates,
        store
    )

    # CONSOLIDATION FIX: same as above -- written and demonstrated in
    # the original notebook, never wired into analyze_store itself.
    validate_kiosk_compactness(
        kiosk_clusters,
        max_cluster_radius_m=200
    )

    kiosk_plan = allocate_media(
        dataframe=kiosk_clusters,
        opportunity_column="kiosk_cluster_index",
        total_units=store.pole_kiosks,
        media_type="pole_kiosks"
    )

    # CONSOLIDATION FIX: short plain-language reasoning per cluster.
    kiosk_plan["why"] = explain_signal_drivers(
        kiosk_plan,
        {
            "mean_commercial_signal": "commercial activity",
            "mean_movement_signal": "movement & transit",
            "mean_trip_generator_signal": "footfall drivers"
        }
    )

    # -----------------------------------------------------
    # 14. No Parking Boards
    # -----------------------------------------------------
    #
    # CONSOLIDATION FIX: the original production path only ever
    # scored the 4 catchment rings (build_board_zone_opportunity),
    # even though real candidate-site clustering for boards
    # (prepare/cluster_no_parking_candidates, summarize_board_
    # clusters) was already written and validated elsewhere in the
    # notebook -- it just never got called from here. Boards got a
    # 4-bucket "which ring" answer while Kiosks got real sites.
    # This now uses real candidate clusters, matching Kiosks, and
    # falls back to the zone-level view only if no discrete site
    # clears the evidence threshold anywhere (e.g. very sparse
    # surroundings).

    board_candidates = prepare_no_parking_candidates(
        analytical_evidence
    )

    if not board_candidates.empty:

        board_candidates = cluster_no_parking_candidates(
            board_candidates,
            store,
            cluster_radius_m=150
        )

        board_clusters = summarize_board_clusters(
            board_candidates,
            store
        )

        validate_board_compactness(
            board_clusters,
            max_cluster_radius_m=200
        )

        board_plan = allocate_media(
            dataframe=board_clusters,
            opportunity_column="board_cluster_index",
            total_units=store.no_parking_boards,
            media_type="no_parking_boards"
        )

        board_placement_method = "candidate_clusters"

    else:

        board_zones = build_board_zone_opportunity(
            zone_density,
            store
        )

        board_plan = allocate_media(
            dataframe=board_zones,
            opportunity_column="board_zone_index",
            total_units=store.no_parking_boards,
            media_type="no_parking_boards"
        )

        board_placement_method = "catchment_zone_fallback"

    # CONSOLIDATION FIX: short plain-language reasoning, branched on
    # which placement method actually ran (see fix #1 above).
    if board_placement_method == "candidate_clusters":
        board_plan["why"] = explain_signal_drivers(
            board_plan,
            {
                "mean_commercial_signal": "commercial activity",
                "mean_movement_signal": "movement & transit",
                "mean_trip_generator_signal": "footfall drivers"
            }
        )
    else:
        board_plan["why"] = explain_signal_drivers(
            board_plan,
            {
                "commercial_density": "commercial activity",
                "movement_density": "movement & transit",
                "trip_generator_density": "footfall drivers"
            }
        )

    # -----------------------------------------------------
    # 15. Final validation
    # -----------------------------------------------------

    validate_final_store_plan(
        store,
        auto_plan,
        kiosk_plan,
        board_plan
    )

    # -----------------------------------------------------
    # 16. Complete result
    # -----------------------------------------------------

    return {
        "store": store,
        "raw_osm": raw_osm,
        "prepared_features": prepared_features,
        "enriched_features": enriched_features,
        "analytical_evidence": analytical_evidence,
        "catchments": catchments,
        "zoned_features": zoned_features,
        "zone_density": zone_density,
        "graph": graph,
        "nodes": node_intelligence,
        "segments": segment_intelligence,
        "operational_segments": operational_segments,
        "corridor_opportunity": corridor_opportunity,
        "auto_plan": auto_plan,
        "kiosk_plan": kiosk_plan,
        "board_plan": board_plan,
        "board_placement_method": board_placement_method,
        "status": "validated"
    }

def build_management_store_report(
    result
):
    """
    Convert a validated Tom and Jerry result into a
    compact management-facing report.

    Internal model objects remain in `result`.
    This function exposes only the decision layer.
    """

    if result is None:
        raise ValueError(
            "result cannot be None."
        )

    if result.get("status") != "validated":
        raise AssertionError(
            "Only validated store results can be "
            "converted into a management report."
        )

    store = result["store"]

    # -----------------------------------------------------
    # Package
    # -----------------------------------------------------

    package = {
        "auto_tops": store.auto_tops,
        "pole_kiosks": store.pole_kiosks,
        "no_parking_boards": store.no_parking_boards
    }

    # -----------------------------------------------------
    # Auto Tops
    # -----------------------------------------------------

    # CONSOLIDATION FIX: shows every corridor that received an
    # allocation, not just a top-N slice -- the full package is
    # always spread across all of these, never just a handful.
    auto = (
        result["auto_plan"]
        .query(
            "auto_tops_allocation > 0"
        )
        .sort_values(
            [
                "corridor_auto_index",
                "auto_tops_allocation"
            ],
            ascending=False
        )
        .copy()
    )

    auto_count = len(auto)

    auto_report = auto[
        [
            "operational_corridor_id",
            "corridor_auto_index",
            "why",
            "auto_tops_allocation",
            "auto_tops_allocation_share"
        ]
    ].reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Pole Kiosks
    # -----------------------------------------------------

    kiosk = (
        result["kiosk_plan"]
        .query(
            "pole_kiosks_allocation > 0"
        )
        .sort_values(
            [
                "kiosk_cluster_index",
                "pole_kiosks_allocation"
            ],
            ascending=False
        )
        .copy()
    )

    kiosk_count = len(kiosk)

    kiosk_report = kiosk[
        [
            "kiosk_cluster_id",
            "kiosk_cluster_index",
            "why",
            "pole_kiosks_allocation",
            "pole_kiosks_allocation_share"
        ]
    ].reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # No Parking Boards
    # -----------------------------------------------------
    #
    # CONSOLIDATION FIX: board_plan now has two possible shapes
    # depending on board_placement_method -- real candidate clusters
    # (board_cluster_id / board_cluster_index) in the normal case, or
    # the catchment-ring fallback (zone / board_zone_index) when no
    # candidate site cleared the evidence threshold. Branch on it
    # rather than assuming the zone-only shape everywhere.

    board_placement_method = result.get(
        "board_placement_method",
        "catchment_zone_fallback"
    )

    if board_placement_method == "candidate_clusters":

        board_id_column = "board_cluster_id"
        board_index_column = "board_cluster_index"

    else:

        board_id_column = "zone"
        board_index_column = "board_zone_index"

    boards = (
        result["board_plan"]
        .query(
            "no_parking_boards_allocation > 0"
        )
        .sort_values(
            [
                board_index_column,
                "no_parking_boards_allocation"
            ],
            ascending=False
        )
        .copy()
    )

    board_count = len(boards)

    board_report = boards[
        [
            board_id_column,
            board_index_column,
            "why",
            "no_parking_boards_allocation",
            "no_parking_boards_allocation_share"
        ]
    ].reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Spatial scale
    # -----------------------------------------------------

    report = {
        "store": {
            "city": store.city,
            "latitude": store.latitude,
            "longitude": store.longitude
        },

        "package": package,

        "data_coverage": {
            "osm_features": len(
                result["raw_osm"]
            ),
            "analytical_evidence": len(
                result["analytical_evidence"]
            ),
            "road_nodes": len(
                result["nodes"]
            ),
            "road_segments": len(
                result["segments"]
            )
        },

        "recommendations": {
            "auto_tops": auto_report,
            "pole_kiosks": kiosk_report,
            "no_parking_boards": board_report
        },

        "location_counts": {
            "auto_tops": auto_count,
            "pole_kiosks": kiosk_count,
            "no_parking_boards": board_count
        },

        "no_parking_boards_placement_method": board_placement_method,

        "status": result["status"]
    }

    return report

def run_store(
    city,
    latitude,
    longitude,
    auto_tops,
    pole_kiosks,
    no_parking_boards,
    radius_m=3000,
    return_full_result=False
):
    """
    Final user-facing Tom and Jerry interface.

    INPUT:
        Store location + media package.

    OUTPUT:
        Management-ready recommendation report.

    Set return_full_result=True when the underlying
    analytical objects are required for audit/debugging.
    """

    # -----------------------------------------------------
    # Run complete analytical engine
    # -----------------------------------------------------

    full_result = analyze_store(
        city=city,
        latitude=latitude,
        longitude=longitude,
        auto_tops=auto_tops,
        pole_kiosks=pole_kiosks,
        no_parking_boards=no_parking_boards,
        radius_m=radius_m
    )

    # -----------------------------------------------------
    # Build compact management report
    # -----------------------------------------------------

    management_report = (
        build_management_store_report(
            full_result
        )
    )

    # -----------------------------------------------------
    # Return requested level of detail
    # -----------------------------------------------------

    if return_full_result:

        return {
            "management_report": management_report,
            "full_result": full_result
        }

    return management_report

def build_store_map(full_result, zoom_start=14):
    """
    Build an interactive map of a validated Tom and Jerry result:
    catchment rings, every allocated Auto Top corridor, Pole Kiosk
    cluster, and No Parking Board site -- each media type in its own
    toggleable layer, and the initial view auto-fit to whatever's
    plotted so nothing sits off-screen.

    Auto Top corridors are each given a distinct colour from a
    qualitative palette (instead of one colour with varying opacity)
    so individual corridors stay visually distinguishable even when
    several run close together on the same road grid -- weight still
    scales with allocation, colour now identifies which corridor is
    which.

    CONSOLIDATION ADDITION: folium was pip-installed in the original
    notebook but never actually used anywhere in it. This is that.

    Pass the FULL result (run_store(..., return_full_result=True)
    -> result["full_result"], or analyze_store(...) directly).
    """

    import folium

    store = full_result["store"]

    m = folium.Map(
        location=[store.latitude, store.longitude],
        zoom_start=zoom_start,
        tiles="cartodbpositron"
    )

    all_points = [[store.latitude, store.longitude]]

    # -----------------------------------------------------
    # Store marker (always on, not part of a toggle layer)
    # -----------------------------------------------------

    folium.Marker(
        [store.latitude, store.longitude],
        popup=f"<b>{store.city}</b><br>Store location",
        tooltip=store.city,
        icon=folium.Icon(color="black", icon="star")
    ).add_to(m)

    # -----------------------------------------------------
    # Catchment rings -- own layer, OFF by default. Reference
    # geography, not a recommendation.
    # -----------------------------------------------------

    catchment_layer = folium.FeatureGroup(
        name="\u26aa Catchment rings (Z1-Z4)", show=False
    )

    catchments = full_result["catchments"].sort_values(
        "outer_radius_m", ascending=False
    )

    for _, row in catchments.iterrows():
        folium.Circle(
            [store.latitude, store.longitude],
            radius=row["outer_radius_m"],
            color="#999999",
            weight=1,
            fill=False,
            dash_array="4,6",
            popup=(
                f"{row['zone']} boundary "
                f"({row['inner_radius_m']:.0f}-{row['outer_radius_m']:.0f}m)"
            )
        ).add_to(catchment_layer)

    catchment_layer.add_to(m)

    # -----------------------------------------------------
    # Auto Top corridors -- every corridor that got an allocation,
    # each drawn in a distinct colour from a warm qualitative
    # palette (kept in the orange/red/amber family so the layer as
    # a whole still reads as "Auto Tops" next to the blue kiosks and
    # green boards, while individual corridors stay tellable apart).
    # -----------------------------------------------------

    CORRIDOR_COLORS = [
        "#e6550d",  # orange
        "#c0392b",  # brick red
        "#e67e22",  # amber orange
        "#943126",  # dark red
        "#d35400",  # burnt orange
        "#a04000",  # dark rust
        "#f39c12",  # gold
        "#7b241c",  # deep maroon
        "#ca6f1e",  # ochre
        "#e74c3c",  # red
        "#b9770e",  # mustard
        "#af601a",  # tan rust
    ]

    auto_plan = full_result["auto_plan"]
    allocated_corridors = (
        auto_plan[auto_plan["auto_tops_allocation"] > 0]
        .sort_values(
            ["corridor_auto_index", "auto_tops_allocation"],
            ascending=False
        )
        .reset_index(drop=True)
    )

    corridor_layer = folium.FeatureGroup(
        name=(
            f"\U0001f7e0 Auto Top corridors "
            f"({len(allocated_corridors)})"
        )
    )

    if not allocated_corridors.empty:

        segments = full_result["operational_segments"]
        max_alloc = allocated_corridors["auto_tops_allocation"].max()

        for rank, corridor in allocated_corridors.iterrows():

            corridor_id = corridor["operational_corridor_id"]
            alloc = corridor["auto_tops_allocation"]
            share = corridor["auto_tops_allocation_share"]
            intensity = (alloc / max_alloc) if max_alloc > 0 else 0.0
            corridor_color = CORRIDOR_COLORS[rank % len(CORRIDOR_COLORS)]

            corridor_segments = segments[
                segments["operational_corridor_id"] == corridor_id
            ]

            for _, seg in corridor_segments.iterrows():

                geom = seg.get("geometry")

                if geom is None or geom.is_empty:
                    continue

                line_parts = (
                    [geom] if geom.geom_type == "LineString"
                    else list(geom.geoms) if geom.geom_type == "MultiLineString"
                    else []
                )

                for part in line_parts:
                    coords = [(lat, lon) for lon, lat in part.coords]
                    all_points.extend([list(c) for c in coords])
                    folium.PolyLine(
                        coords,
                        color=corridor_color,
                        weight=1.5 + 2 * intensity,
                        opacity=0.75,
                        popup=(
                            f"Corridor {corridor_id}: {alloc} Auto Tops "
                            f"({share:.0%} of package)<br>"
                            f"{corridor['why']}"
                        )
                    ).add_to(corridor_layer)

    corridor_layer.add_to(m)

    # -----------------------------------------------------
    # Pole Kiosk clusters -- every cluster that got an allocation
    # -----------------------------------------------------

    kiosk_plan = full_result["kiosk_plan"]
    allocated_kiosks = (
        kiosk_plan[kiosk_plan["pole_kiosks_allocation"] > 0]
        .sort_values(
            ["kiosk_cluster_index", "pole_kiosks_allocation"],
            ascending=False
        )
    )

    kiosk_layer = folium.FeatureGroup(
        name=(
            f"\U0001f535 Pole Kiosk clusters "
            f"({len(allocated_kiosks)})"
        )
    )

    for _, row in allocated_kiosks.iterrows():
        all_points.append(
            [row["cluster_latitude"], row["cluster_longitude"]]
        )
        folium.CircleMarker(
            [row["cluster_latitude"], row["cluster_longitude"]],
            radius=9 + row["pole_kiosks_allocation"] ** 0.5,
            color="#08519c",
            weight=2,
            fill=True,
            fill_color="#3182bd",
            fill_opacity=0.9,
            popup=(
                f"Kiosk cluster {row['kiosk_cluster_id']}: "
                f"{row['pole_kiosks_allocation']} Pole Kiosks "
                f"({row['pole_kiosks_allocation_share']:.0%})<br>"
                f"{row['why']}"
            )
        ).add_to(kiosk_layer)

    kiosk_layer.add_to(m)

    # -----------------------------------------------------
    # No Parking Boards -- every site that got an allocation.
    # Candidate clusters (points) or catchment-zone rings on the
    # sparse-area fallback.
    # -----------------------------------------------------

    board_plan = full_result["board_plan"]
    method = full_result.get(
        "board_placement_method", "catchment_zone_fallback"
    )
    board_index_column = (
        "board_cluster_index" if method == "candidate_clusters"
        else "board_zone_index"
    )
    allocated_boards = (
        board_plan[board_plan["no_parking_boards_allocation"] > 0]
        .sort_values(
            [board_index_column, "no_parking_boards_allocation"],
            ascending=False
        )
    )

    board_layer = folium.FeatureGroup(
        name=(
            f"\U0001f7e2 No Parking Boards "
            f"({len(allocated_boards)})"
        )
    )

    if method == "candidate_clusters":

        for _, row in allocated_boards.iterrows():
            all_points.append(
                [row["cluster_latitude"], row["cluster_longitude"]]
            )
            folium.CircleMarker(
                [row["cluster_latitude"], row["cluster_longitude"]],
                radius=8 + row["no_parking_boards_allocation"] ** 0.5,
                color="#006d2c",
                weight=2,
                fill=True,
                fill_color="#31a354",
                fill_opacity=0.9,
                popup=(
                    f"Board cluster {row['board_cluster_id']}: "
                    f"{row['no_parking_boards_allocation']} No Parking Boards "
                    f"({row['no_parking_boards_allocation_share']:.0%})<br>"
                    f"{row['why']}"
                )
            ).add_to(board_layer)

    else:

        zone_bounds = full_result["catchments"].set_index("zone")

        for _, row in allocated_boards.iterrows():
            zone = row["zone"]
            if zone not in zone_bounds.index:
                continue
            outer_r = zone_bounds.loc[zone, "outer_radius_m"]
            all_points.append(
                [store.latitude + outer_r / 111_000, store.longitude]
            )
            folium.Circle(
                [store.latitude, store.longitude],
                radius=outer_r,
                color="#006d2c",
                weight=2,
                fill=True,
                fill_color="#31a354",
                fill_opacity=0.18,
                popup=(
                    f"{zone}: {row['no_parking_boards_allocation']} "
                    f"No Parking Boards "
                    f"({row['no_parking_boards_allocation_share']:.0%})<br>"
                    f"{row['why']}"
                )
            ).add_to(board_layer)

    board_layer.add_to(m)

    # -----------------------------------------------------
    # Layer control -- toggle any group on/off, useful now that
    # every allocated location is drawn rather than just a top 5.
    # -----------------------------------------------------

    folium.LayerControl(collapsed=False).add_to(m)

    # -----------------------------------------------------
    # Auto-fit the initial view to everything actually plotted.
    # -----------------------------------------------------

    if len(all_points) > 1:
        lats = [p[0] for p in all_points]
        lons = [p[1] for p in all_points]
        m.fit_bounds(
            [[min(lats), min(lons)], [max(lats), max(lons)]],
            padding=(30, 30)
        )

    return m


def build_store_onepager(result):
    """
    Combine the management report and the interactive map into one
    self-contained HTML file per store -- handoff-ready for a
    Regional Manager, rather than living only inside a Colab cell.

    Pass the full return value of run_store(..., return_full_result=True).
    """

    import datetime

    report = result["management_report"]
    full = result["full_result"]

    store = report["store"]
    package = report["package"]
    counts = report["location_counts"]

    # -----------------------------------------------------
    # STEP A: turn each report table into a styled HTML table
    # (not a raw pandas dump)
    # -----------------------------------------------------

    def render_table(df):
        if df.empty:
            return "<p class='empty'>No locations recommended.</p>"
        return df.to_html(index=False, classes="rec-table", border=0, escape=False)

    auto_table_html = render_table(report["recommendations"]["auto_tops"])
    kiosk_table_html = render_table(report["recommendations"]["pole_kiosks"])
    board_table_html = render_table(report["recommendations"]["no_parking_boards"])

    # -----------------------------------------------------
    # STEP B: build the interactive map and grab its embeddable HTML
    # -----------------------------------------------------

    store_map = build_store_map(full)
    map_embed_html = store_map._repr_html_()

    # -----------------------------------------------------
    # STEP C: page styling (kept in one place, one CSS block)
    # -----------------------------------------------------

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
           margin: 0; padding: 0; background: #f7f7f8; color: #222; }
    .header { background: #1a1a2e; color: white; padding: 28px 40px; }
    .header h1 { margin: 0 0 6px 0; font-size: 24px; }
    .header .meta { color: #b8b8c8; font-size: 13px; }
    .package-bar { display: flex; gap: 24px; background: #16213e; padding: 14px 40px;
                   color: white; font-size: 14px; }
    .package-bar b { font-size: 18px; display: block; }
    .content { padding: 30px 40px; }
    .section { background: white; border-radius: 8px; padding: 22px 26px;
               margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    .section h2 { margin-top: 0; font-size: 17px; color: #1a1a2e; }
    table.rec-table { border-collapse: collapse; width: 100%; font-size: 13px; }
    table.rec-table th { text-align: left; background: #f0f0f4; padding: 8px 10px;
                          border-bottom: 2px solid #ddd; }
    table.rec-table td { padding: 7px 10px; border-bottom: 1px solid #eee; }
    .empty { color: #888; font-style: italic; }
    .footer { padding: 16px 40px; color: #888; font-size: 12px; }
    """

    # -----------------------------------------------------
    # STEP D: assemble the full page
    # -----------------------------------------------------

    generated_at = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{store['city']} -- OOH Siting Plan</title>
<style>{css}</style>
</head>
<body>

<div class="header">
    <h1>{store['city']} -- OOH Siting Plan</h1>
    <div class="meta">Generated {generated_at} &middot; {store['latitude']:.4f}, {store['longitude']:.4f}</div>
</div>

<div class="package-bar">
    <div><b>{package['auto_tops']}</b>Auto Tops</div>
    <div><b>{package['pole_kiosks']}</b>Pole Kiosks</div>
    <div><b>{package['no_parking_boards']}</b>No Parking Boards</div>
</div>

<div class="content">

    <div class="section">
        <h2>Auto Top Corridors ({counts['auto_tops']})</h2>
        {auto_table_html}
    </div>

    <div class="section">
        <h2>Pole Kiosk Clusters ({counts['pole_kiosks']})</h2>
        {kiosk_table_html}
    </div>

    <div class="section">
        <h2>No Parking Board Sites ({counts['no_parking_boards']})</h2>
        {board_table_html}
    </div>

    <div class="section">
        <h2>Map</h2>
        {map_embed_html}
    </div>

</div>

<div class="footer">
    Retail OOH Siting Engine &middot; built from public OpenStreetMap data.
    Not empirically calibrated against footfall or redemption data -- treat scores as a planning
    priority signal, not a guarantee.
</div>

</body>
</html>"""

    return html
def build_vendor_sheet(result):
    """
    A flat, no-frills site list for the vendor/agency doing the
    physical installation -- just where each unit goes and how many,
    with a clickable Google Maps link. No index scores, no "why"
    reasoning -- that's for the management one-pager, not this.

    Returns a pandas DataFrame; convert with .to_csv(index=False)
    for a downloadable file.
    """

    full = result["full_result"]
    store = full["store"]
    rows = []

    # -----------------------------------------------------
    # Auto Tops -- corridor midpoint as the representative point
    # (it's a route, not a single site, so this is "centre of the
    # corridor" rather than an exact install location)
    # -----------------------------------------------------

    auto_plan = full["auto_plan"]
    allocated_corridors = auto_plan[auto_plan["auto_tops_allocation"] > 0]
    segments = full["operational_segments"]

    for _, corridor in allocated_corridors.iterrows():
        corridor_id = corridor["operational_corridor_id"]
        corridor_segments = segments[
            segments["operational_corridor_id"] == corridor_id
        ]
        # midpoint of the corridor's longest segment as the reference point
        if not corridor_segments.empty:
            longest = corridor_segments.loc[corridor_segments["length"].idxmax()]
            mid_point = longest["geometry"].interpolate(0.5, normalized=True)
            lat, lon = mid_point.y, mid_point.x
        else:
            lat, lon = store.latitude, store.longitude

        rows.append({
            "Media Type": "Auto Top",
            "Site ID": f"Corridor {corridor_id}",
            "Quantity": int(corridor["auto_tops_allocation"]),
            "Latitude": round(lat, 6),
            "Longitude": round(lon, 6),
            "Google Maps Link": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
            "Notes": "Route/corridor -- distribute along this road, not a single point"
        })

    # -----------------------------------------------------
    # Pole Kiosks -- exact cluster point
    # -----------------------------------------------------

    kiosk_plan = full["kiosk_plan"]
    for _, row in kiosk_plan[kiosk_plan["pole_kiosks_allocation"] > 0].iterrows():
        lat, lon = row["cluster_latitude"], row["cluster_longitude"]
        rows.append({
            "Media Type": "Pole Kiosk",
            "Site ID": f"Kiosk cluster {row['kiosk_cluster_id']}",
            "Quantity": int(row["pole_kiosks_allocation"]),
            "Latitude": round(lat, 6),
            "Longitude": round(lon, 6),
            "Google Maps Link": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
            "Notes": ""
        })

    # -----------------------------------------------------
    # No Parking Boards -- cluster point, or zone centre on fallback
    # -----------------------------------------------------

    board_plan = full["board_plan"]
    method = full.get("board_placement_method", "catchment_zone_fallback")
    allocated_boards = board_plan[board_plan["no_parking_boards_allocation"] > 0]

    if method == "candidate_clusters":
        for _, row in allocated_boards.iterrows():
            lat, lon = row["cluster_latitude"], row["cluster_longitude"]
            rows.append({
                "Media Type": "No Parking Board",
                "Site ID": f"Board cluster {row['board_cluster_id']}",
                "Quantity": int(row["no_parking_boards_allocation"]),
                "Latitude": round(lat, 6),
                "Longitude": round(lon, 6),
                "Google Maps Link": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
                "Notes": ""
            })
    else:
        for _, row in allocated_boards.iterrows():
            rows.append({
                "Media Type": "No Parking Board",
                "Site ID": f"Zone {row['zone']}",
                "Quantity": int(row["no_parking_boards_allocation"]),
                "Latitude": round(store.latitude, 6),
                "Longitude": round(store.longitude, 6),
                "Google Maps Link": f"https://www.google.com/maps?q={store.latitude:.6f},{store.longitude:.6f}",
                "Notes": f"Sparse-area fallback -- distribute across the {row['zone']} ring around the store, not one fixed point"
            })

    return pd.DataFrame(rows)
