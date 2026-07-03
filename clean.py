from fetch import fetch_worldcup


def format_goal(goal):
    name = goal["name"]
    minute = goal["minute"]
    final = f"{name} {minute}'"
    if goal.get("penalty"):
        final += " (pen)"
    return final


def is_placeholder(team):
    return any(c.isdigit() for c in team)


def clean_match(match):
    home = match["team1"]
    away = match["team2"]
    date = match.get("date")

    if is_placeholder(home) or is_placeholder(away):
        return None

    score = match.get("score")
    if score is None:
        status = "scheduled"
        home_goals = None
        away_goals = None
        ht_home = None
        ht_away = None
        result_note = None
    else:
        status = "played"
        open_play = score.get("et") or score.get("ft")
        home_goals, away_goals = open_play
        ht = score.get("ht")
        ht_home, ht_away = ht if ht else (None, None)

        result_note = None
        if "p" in score:
            p = score["p"]
            winner = home if p[0] > p[1] else away
            result_note = f"{winner} won {max(p)}-{min(p)} on penalties"
        elif "et" in score:
            result_note = "decided after extra time"

    home_scorers = [
        f"{home}: {format_goal(g)}" for g in match.get("goals1", [])]
    away_scorers = [
        f"{away}: {format_goal(g)}" for g in match.get("goals2", [])]
    scorers = home_scorers + away_scorers
    return {
        "match_id": f"{date}-{home}-{away}",
        "stage": match.get("group"),
        "round": match.get("round"),
        "date": date,
        "home": home,
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "ht_home": ht_home,
        "ht_away": ht_away,
        "result_note": result_note,
        "scorers": scorers,
        "venue": match.get("ground"),
        "status": status,
    }


def clean_matches(raw):
    records = []
    for match in raw["matches"]:
        record = clean_match(match)
        if record is not None:
            records.append(record)
    return records


if __name__ == "__main__":
    raw = fetch_worldcup(2026)
    records = clean_matches(raw)
    print(f"cleaned {len(records)} matches")
    for r in records[:3]:
        print(r)
