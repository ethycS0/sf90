import requests

url = "https://api.similarweb.com/v3/batch/traffic_and_engagement/request-report"

payload = {
    "metrics": ["all_traffic_visits", "global_rank", "desktop_new_visitors", "mobile_average_visit_duration"],
    "filters": {
        "domains": ["facebook.com"],
        "countries": ["WW"],
        "include_subdomains": True
    },
    "granularity": "monthly",
    "start_date": "2022-06",
    "end_date": "2023-06",
    "response_format": "csv",
    "delivery_method": "download_link"
}
headers = {
    "accept": "application/json",
    "api-key": "Batch API Key",
    "content-type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)