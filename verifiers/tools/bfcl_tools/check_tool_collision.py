from huanzhi_utils import load_file

FILE_PATH = ["/home/richard/verifiers/verifiers/tools/bfcl_tools/gorilla_file_system.json",
             "/home/richard/verifiers/verifiers/tools/bfcl_tools/math_api.json",
             "/home/richard/verifiers/verifiers/tools/bfcl_tools/message_api.json",
             "/home/richard/verifiers/verifiers/tools/bfcl_tools/posting_api.json",
             "/home/richard/verifiers/verifiers/tools/bfcl_tools/ticket_api.json",
             "/home/richard/verifiers/verifiers/tools/bfcl_tools/trading_bot.json",
             "/home/richard/verifiers/verifiers/tools/bfcl_tools/travel_booking.json",
             "/home/richard/verifiers/verifiers/tools/bfcl_tools/vehicle_control.json",
             ]

all_tools = []
for file_path in FILE_PATH:
    file_content = load_file(file_path)
    all_tools.extend(file_content)

# Check if any collided "name" in all_tools
for tool in all_tools:
    if all_tools.count(tool["name"]) > 1:
        print(f"Collision found: {tool['name']}")
print("No collision found")