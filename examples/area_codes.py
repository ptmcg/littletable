"""
What do telephone cords have to do with area codes?

Telephone cords always had 4 wires: red, green, black, and white. But for
most house and office phones back in the day, all you needed to connect were
the red and green ones.

In analog times, "dialing" each number involved opening and closing the circuit
N number of times (to dial a "0" required 10 circuit open/closes). Rotary phones
did this for you, but you could do it yourself by tapping the wire on the
mounting screw (I think this was actually used as an escape in a 70's era spy
movie or TV show).

Phone exchanges did the same thing with physical switches to open and close
circuits. When area codes were added, their 3-digit pattern had to be detectable
as different from regular phone numbers. The middle number had to be a 0 or 1,
and the first and third numbers had to be digits 2-9. Since the switches were
physical, area codes were assigned by number and activity of phones, so that the
most phone-active areas used the lowest possible numbers for the fewest number
 of circuit open/close sequences. So who got 212? New York City (Manhattan).
213? Los Angeles. 312? Chicago. 214? Dallas. 412? Pittsburgh.

1947 map of initial area codes:
https://upload.wikimedia.org/wikipedia/en/thumb/7/72/North_American_Numbering_Plan_NPA_map_BTM_1947.png/1400px-North_American_Numbering_Plan_NPA_map_BTM_1947.png

Now that the phone system is completely digital, these area code constraints
have largely been lifted (still no area codes start with 0 or 1, and the x00 and
x11 codes are reserved for special uses).
"""
import littletable as lt
from typing import Union

try:
    area_codes = lt.csv_import(
        "area_codes.csv",
        transforms = {"latitude": float, "longitude": float},
    )
except FileNotFoundError:
    area_codes = lt.csv_import(
        "https://raw.githubusercontent.com/ravisorg/Area-Code-Geolocation-Database/refs/heads/master/us-area-code-cities.csv",
        fieldnames = "area_code,city,state,country,latitude,longitude".split(","),
        transforms = {"latitude": float, "longitude": float},
    )
    area_codes.csv_export("area_codes.csv")

# collapse duplicate entries for the same area code
area_codes = area_codes.unique("area_code")
print(len(area_codes))


class AreaCode:
    def __init__(self, digits: Union[tuple[int, int, int], str]):
        if isinstance(digits, str):
            digits = tuple(int(d) for d in digits)
        self.digits: tuple[int, int, int] = digits

    @property
    def is_analog(self):
        return self.digits[1] in (0, 1) and not set(self.digits[::2]) & {0, 1}

    def __len__(self):
        return sum(self.digits) + (10 * (self.digits.count(0)))

    def __str__(self):
        return "".join(str(d) for d in self.digits)


area_codes.compute_field("area_code", lambda rec: AreaCode(rec.area_code))

# analog area codes must start and end with digit 2-9, and have middle digit of 0 or 1
analog_codes: lt.Table = area_codes.where(lambda rec: rec.area_code.is_analog)
print(len(analog_codes))

# find those area codes that require the fewest clicks
analog_codes.sort(key=lambda rec: len(rec.area_code))

analog_codes.compute_field("clicks", lambda rec: len(rec.area_code))
analog_codes.where(clicks=lt.Table.le(14)).present(width=120)
