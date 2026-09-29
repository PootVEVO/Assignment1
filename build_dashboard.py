import json
from pathlib import Path

# Read the dashboard template
template_path = Path("/Users/iananderson/Documents/Assign1/dashboard_template.html")
html_content = template_path.read_text(encoding="utf-8")

# Read the dashboard data
with open("/Users/iananderson/Documents/Assign1/dashboard_data.json") as f:
    data = json.load(f)

# Replace the placeholder with actual data as a properly encoded string
data_json = json.dumps(data, separators=(',', ':'))
data_encoded = data_json.replace('%', '%25').replace('"', '%22')
html_content = html_content.replace('JSON.parse(decodeURIComponent("%DASHBOARD_DATA_PLACEHOLDER%"))', f'JSON.parse("{data_encoded}")')

# Write the final dashboard
final_path = Path("/Users/iananderson/Documents/Assign1/dashboard.html")
final_path.write_text(html_content, encoding="utf-8")

print(f"Dashboard written: {final_path}")
print(f"Size: {final_path.stat().st_size} bytes")
