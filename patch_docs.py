import sys

filepath = 'pyair2stream/cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

content = content.replace("See CROSS_VALIDATION_PLAN.md for full rationale. Summary of the design:", "Summary of the design:")
content = content.replace("See module docstring and\n    CROSS_VALIDATION_PLAN.md section 6 for the full rationale.", "See module docstring for the full rationale.")

with open(filepath, 'w') as f:
    f.write(content)
print("Successfully patched documentation references")
