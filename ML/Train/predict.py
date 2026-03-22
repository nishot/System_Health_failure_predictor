def get_risk_level(prob):
    if prob < 0.3:
        return "LOW"
    elif prob <0.7:
        return "MEDIUM"
    else:
        return "HIGH"
    


def predict_failure(model,input_data):
    prob=model.predict_proba([input_data])[0][1]
    risk=get_risk_level(prob)

    return {
        "failure_probability":round(prob,3),
        "risk_level":risk
    }

