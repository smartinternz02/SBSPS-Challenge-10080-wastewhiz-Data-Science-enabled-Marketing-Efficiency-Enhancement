from django.shortcuts import render
import pickle
import pandas as pd
import numpy as np


def home(request):
    return render(request, "index.html")


def prepare(
    industry,
    deal_value,
    weighted_amt,
    pitch,
    revenue,
    fund,
    geo,
    location,
    desgn,
    hcr,
    source,
    level,
    update,
    resource,
    rating,
):
    train = pd.read_csv(r"C:\Users\Ananth\Downloads\train.csv")

    train2 = train.drop(
        [
            "Date_of_creation",
            "Contact_no",
            "POC_name",
            "Lead_POC_email",
            "Internal_POC",
            "Deal_title",
            "Lead_name",
        ],
        axis=1,
    )

    data = {
        "Industry": industry,
        "Deal_value": deal_value,
        "Weighted_amount": weighted_amt,
        "Pitch": pitch,
        "Lead_revenue": revenue,
        "Fund_category": fund,
        "Geography": geo,
        "Location": location,
        "Designation": desgn,
        "Hiring_candidate_role": hcr,
        "Lead_source": source,
        "Level_of_meeting": level,
        "Last_lead_update": update,
        "Resource": resource,
        "Internal_rating": rating,
        "Success_probability": -5,
    }

    train2.loc[len(train2)] = data

    train2["Designation"] = train2["Designation"].replace(
        {
            "Chairman/CEO/President": "Chairman/CEO/President",
            "CEO/Chairman/President": "Chairman/CEO/President",
            "Chief Executive Officer": "CEO",
            "Vice President / GM (04-present) : VP Sales and Marketing (01-04)": "Vice President/GM",
        }
    )

    train2["Last_lead_update"].replace("?", "No track", inplace=True)
    train2["Last_lead_update"].replace(np.nan, "No track", inplace=True)

    train2 = train2.dropna(
        how="any", subset=["Industry", "Resource", "Deal_value", "Location"], axis=0
    )

    train2["Geography"] = train2.apply(
        lambda row: "USA"
        if pd.isna(row["Geography"]) and "," in row["Location"]
        else ("India" if pd.isna(row["Geography"]) else row["Geography"]),
        axis=1,
    )
    train2["Level_of_meeting"] = train2.apply(
        lambda row: 1
        if "1" in row["Level_of_meeting"]
        else (2 if "2" in row["Level_of_meeting"] else 3),
        axis=1,
    )

    train2["Deal_value"] = train2["Deal_value"].str.replace("$", "")
    train2["Weighted_amount"] = train2["Weighted_amount"].str.replace("$", "")
    train2["Weighted_amount"] = train2["Weighted_amount"].astype(float)
    train2["Deal_value"] = train2["Deal_value"].astype(float)
    train2["Weighted_amount"].fillna(train2["Weighted_amount"].mean(), inplace=True)

    train2.drop("Location", axis=1, inplace=True)

    train2 = pd.get_dummies(
        train2,
        columns=[
            "Industry",
            "Pitch",
            "Lead_revenue",
            "Fund_category",
            "Geography",
            "Designation",
            "Hiring_candidate_role",
            "Lead_source",
            "Last_lead_update",
            "Resource",
        ],
    )

    # print(train2.head())
    # print(list(train2.iloc[-1, :-1]))

    return list(train2.iloc[-1, :-1])


def getPredictions(
    industry,
    deal_value,
    weighted_amt,
    pitch,
    revenue,
    fund,
    geo,
    location,
    desgn,
    hcr,
    source,
    level,
    update,
    resource,
    rating,
):
    model = pickle.load(open("ml_model.sav", "rb"))

    prepared_data = prepare(
        industry,
        deal_value,
        weighted_amt,
        pitch,
        revenue,
        fund,
        geo,
        location,
        desgn,
        hcr,
        source,
        level,
        update,
        resource,
        rating,
    )

    prediction = model.predict([prepared_data])

    return prediction


def result(request):
    industry = request.GET["industry"]
    deal_value = request.GET["deal_value"]
    weighted_amt = request.GET["weighted_amt"]
    pitch = request.GET["pitch"]
    revenue = request.GET["revenue"]
    fund = request.GET["fund"]
    geo = request.GET["geo"]
    location = request.GET["location"]
    desgn = request.GET["desgn"]
    hcr = request.GET["hcr"]
    source = request.GET["source"]
    level = request.GET["level"]
    update = request.GET["update"]
    resource = request.GET["resource"]
    rating = int(request.GET["rating"])

    result = getPredictions(
        industry,
        deal_value,
        weighted_amt,
        pitch,
        revenue,
        fund,
        geo,
        location,
        desgn,
        hcr,
        source,
        level,
        update,
        resource,
        rating,
    )

    return render(request, "result.html", {"result": result})
