"""API REST (FastAPI) para clasificar variedades de frijol seco."""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from . import __version__, config, predict as pred

# Modelo Pydantic generado desde config.RANGES (sin números mágicos)
_fields = {
    name: (float, Field(..., ge=lo, le=hi, description=f"Rango válido [{lo}, {hi}]"))
    for name, (lo, hi) in config.RANGES.items()
}
Bean = create_model("Bean", __config__=ConfigDict(extra="forbid", json_schema_extra={
    "example": {
        "Area": 28395, "Perimeter": 610.291, "MajorAxisLength": 208.178,
        "MinorAxisLength": 173.889, "AspectRation": 1.197, "Eccentricity": 0.55,
        "ConvexArea": 28715, "EquivDiameter": 190.141, "Extent": 0.764,
        "Solidity": 0.989, "roundness": 0.958, "Compactness": 0.913,
        "ShapeFactor1": 0.00733, "ShapeFactor2": 0.00314,
        "ShapeFactor3": 0.834, "ShapeFactor4": 0.999}}), **_fields)


class Prediction(BaseModel):
    variety: str
    confidence: float
    probabilities: dict[str, float]


app = FastAPI(title="Dry Bean API", version=__version__,
              description="Clasifica 7 variedades de frijol seco a partir de 16 medidas.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        return pred.model_info()
    except pred.ModelNotFound as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=Prediction)
def predict(bean: Bean):  # type: ignore[valid-type]
    try:
        return pred.predict(bean.model_dump())
    except pred.ModelNotFound as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def main() -> None:
    import uvicorn
    uvicorn.run("drybean_mlops.api:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
