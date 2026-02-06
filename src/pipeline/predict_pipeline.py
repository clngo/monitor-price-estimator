import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict_from_dict(self, payload: dict):
        try:
            df = pd.DataFrame([payload])
            return self.predict(df)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_with_details(self, features):
        try:
            preds = self.predict(features)
            stats_path = "artifacts/price_stats.pkl"
            try:
                stats = load_object(file_path=stats_path)
            except Exception:
                stats = None
            results = []
            for i, pred in enumerate(preds):
                detail = {"prediction": float(pred)}
                if stats:
                    train_prices = stats.get("train_prices")
                    mae = stats.get("mae")
                    if mae is not None:
                        detail["range_low"] = float(pred - mae)
                        detail["range_high"] = float(pred + mae)
                results.append(detail)
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def predict_with_details_from_dict(self, payload: dict):
        try:
            df = pd.DataFrame([payload])
            return self.predict_with_details(df)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try: 
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            if "resolution_width" in features.columns and "resolution_height" in features.columns:
                features = features.copy()
                features["resolution_pixels"] = (
                    features["resolution_width"] * features["resolution_height"]
                )

            if "price_currency" in features.columns:
                features = features.drop(columns=["price_currency"])

            if "shipping_cost" in features.columns:
                features["shipping_cost"] = features["shipping_cost"].clip(lower=0, upper=100)

            drop_optional = [
                "condition_id",
                "seller_feedback_pct",
                "seller_feedback_score",
                "buying_options",
                "shipping_cost_type",
            ]
            drop_optional = [c for c in drop_optional if c in features.columns]
            if drop_optional:
                features = features.drop(columns=drop_optional)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        condition: str | None = None,
        shipping_cost: float | None = None,
        shipping_cost_type: str | None = None,
        brand: str | None = None,
        color: str | None = None,
        screen_size_in: float | None = None,
        resolution_width: float | None = None,
        resolution_height: float | None = None,
        refresh_rate_hz: float | None = None,
        panel_type: str | None = None,
        aspect_ratio: str | None = None,
        response_time_ms: float | None = None,
        hdr: str | None = None,
        has_adaptive_sync: str | None = None,
    ):
        self.condition = condition
        self.shipping_cost = shipping_cost
        self.shipping_cost_type = shipping_cost_type
        self.brand = brand
        self.color = color
        self.screen_size_in = screen_size_in
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.refresh_rate_hz = refresh_rate_hz
        self.panel_type = panel_type
        self.aspect_ratio = aspect_ratio
        self.response_time_ms = response_time_ms
        self.hdr = hdr
        self.has_adaptive_sync = has_adaptive_sync

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "condition": [self.condition],
                "shipping_cost": [self.shipping_cost],
                "shipping_cost_type": [self.shipping_cost_type],
                "brand": [self.brand],
                "color": [self.color],
                "screen_size_in": [self.screen_size_in],
                "resolution_width": [self.resolution_width],
                "resolution_height": [self.resolution_height],
                "refresh_rate_hz": [self.refresh_rate_hz],
                "panel_type": [self.panel_type],
                "aspect_ratio": [self.aspect_ratio],
                "response_time_ms": [self.response_time_ms],
                "hdr": [self.hdr],
                "has_adaptive_sync": [self.has_adaptive_sync],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


def test_predict(payload: dict | None = None):
    try:
        if payload is None:
            payload = {
                "condition": "Certified - Refurbished",
                "shipping_cost": 0,
                "shipping_cost_type": "free",
                "brand": "acer",
                "color": None,
                "screen_size_in": 27,
                "resolution_width": 1920,
                "resolution_height": 1080,
                "refresh_rate_hz": 180,
                "panel_type": "IPS",
                "aspect_ratio": "16:9",
                "response_time_ms": 1,
                "hdr": None,
                "has_adaptive_sync": True,
            }
        return PredictPipeline().predict_with_details_from_dict(payload)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    print(test_predict())   
