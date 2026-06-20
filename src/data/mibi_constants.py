"""Shared MIBI-TNBC label mappings used across scripts and notebooks."""

PATIENT_CLASS_LABELS = {
    0: "mixed",
    1: "compartmentalized",
    2: "cold",
}

GROUP_LABELS = {
    1: "Unidentified",
    2: "Immune",
    3: "Endothelial",
    4: "Mesenchymal-like",
    5: "Tumor",
    6: "Keratin-positive tumor",
}

IMMUNE_GROUP_LABELS = {
    1: "Tregs",
    2: "CD4 T",
    3: "CD8 T",
    4: "CD3 T",
    5: "NK",
    6: "B",
    7: "Neutrophils",
    8: "Macrophages",
    9: "DC",
    10: "DC/Mono",
    11: "Mono/Neu",
    12: "Other immune",
}
