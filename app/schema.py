from pydantic import BaseModel

class systemData(BaseModel):
    Type:int
    Air_temp:float
    Process_temp:float
    Torque:float
    Tool_wear:float
    Rotational_speed:float

