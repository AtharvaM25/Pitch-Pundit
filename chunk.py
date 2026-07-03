from langchain_core.documents import Document

from clean import clean_matches
from fetch import fetch_worldcup


def match_to_document(record):
    stage_value = record["stage"]
    phase = "group" if (
        stage_value and stage_value.startswith("Group")) else "knockout"
    display_stage = stage_value or record["round"]

    if record["status"] == "played":
        h, a = record["home"], record["away"]
        ag, hg = record["away_goals"], record["home_goals"]
        ht_ag, ht_hg = record["ht_away"], record["ht_home"]

        if hg > ag:
            result = f"{h} beat {a} by {hg}-{ag}"
        elif hg < ag:
            result = f"{a} beat {h} by {ag}-{hg}"
        else:
            result = f"{h} drew with {a} {hg}-{ag}"

        parts = [f"{display_stage} match on {record['date']}: {result}"]
        if record["ht_home"] is not None:
            parts.append(f" (half-time {ht_hg}-{ht_ag})")

        parts.append(f" at {record['venue']}. ")
        if record["result_note"]:
            parts.append(record["result_note"] + ".")

        if record["scorers"]:
            parts.append("Scorers — " + ", ".join(record["scorers"]) + ".")

        page_content = " ".join(parts)

    else:
        page_content = f"{display_stage} match on {record['date']}: {record['home']} vs {record['away']} at {record['venue']}"

    metadata = {
        "match_id": record["match_id"],
        "stage": record["stage"],
        "round": record["round"],
        "phase": phase,
        "date": record["date"],
        "home": record["home"],
        "away": record["away"],
        "status": record["status"],
    }

    return Document(page_content=page_content, metadata=metadata)


def records_to_documents(records):
    return [match_to_document(r) for r in records]


if __name__ == "__main__":
    records = clean_matches(fetch_worldcup(2026))
    documents = records_to_documents(records)
    print(f"built {len(documents)} documents\n")

    played = next(d for d in documents if d.metadata["status"] == "played")
    scheduled = next(
        d for d in documents if d.metadata["status"] == "scheduled")
    print("PLAYED:\n", played.page_content, "\n")
    print("SCHEDULED:\n", scheduled.page_content)
