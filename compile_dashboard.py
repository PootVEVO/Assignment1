import json
from pathlib import Path

# Read the dashboard HTML
dashboard_path = Path("/Users/iananderson/Documents/Assign1/dashboard.html")
html_content = dashboard_path.read_text(encoding="utf-8")

# Read the dashboard data
with open("/Users/iananderson/Documents/Assign1/dashboard_data.json") as f:
    data = json.load(f)

# Replace the inline data script
data_json = json.dumps(data, indent=2)
html_content = html_content.replace("const DATA = {};", f"const DATA = {data_json};")

# Also update the review table rows to come from data
html_content = html_content.replace(
    "const reviewRows = [",
    "const reviewRows = DATA.review_rows;"
)

dashboard_path.write_text(html_content, encoding="utf-8")
print("Dashboard HTML updated with data")
