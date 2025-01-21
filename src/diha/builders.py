import json

from diha.calc import ReinforcementConcreteSectionBase
from diha.fibers import RoundFiber
from diha.materials import SteelMaterial, ConcreteMaterial
from diha.sections import RectangularRCSectionBase


class SectionBuilder:

    def __init__(self):
        super().__init__()

    def build(self, section_dict) -> ReinforcementConcreteSectionBase:
        steel = SteelMaterial(**section_dict["steel"])
        concrete = ConcreteMaterial(**section_dict["concrete"])
        bars = [RoundFiber(steel, bar.get("center"), bar.get("diam")) for bar in section_dict["bars"]]
        properties = section_dict.get("properties")
        if section_dict["type"] == "rectangular":
            b = properties.get("b")
            h = properties.get("h")
            return RectangularRCSectionBase(
                concrete, steel, b, h, bars,
                stirrups=properties.get('stirrups', None),
                div_y=properties.get('div_y', None), div_z=properties.get('div_z', None)
            )
        else:
            raise NotImplementedError("La sección %s no está implementado" % section_dict["type"])

    def from_json(self, filename) -> ReinforcementConcreteSectionBase:
        with open(filename, "r") as f:
            return self.build(json.load(f))
