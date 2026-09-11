from typing import TypedDict


class Point(TypedDict):
    """
    A named geographic point used for sampling and map annotations.

    Parameters
    ----------
    name : str
        Human-readable point name.
    lat : float
        Point latitude in degrees north.
    lon : float
        Point longitude in degrees east.
    """

    name: str
    lat: float
    lon: float
