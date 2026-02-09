import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

# === CONFIG ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.infomaniak.com/2/ai/48/openai/v1"
TEACHER_MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 5000
MAX_WORKERS = 5

SYSTEM_PROMPT = """You are a chess grandmaster and teacher. Reason step by step about chess positions, openings and defenses. Always structure your reasoning inside <reasoning>...</reasoning> tags before giving your final answer. Be thorough in your analysis."""

chess_instructions = [
    # === THÉORIE D'OUVERTURE (30) ===
    # Ouvertures blancs
    "Explain the main ideas behind the Italian Game (1.e4 e5 2.Nf3 Nc6 3.Bc4). What are White's key plans?",
    "What is the Ruy Lopez (1.e4 e5 2.Nf3 Nc6 3.Bb5) and why has it been a top opening for centuries?",
    "Explain the Queen's Gambit (1.d4 d5 2.c4). What happens if Black accepts? What if Black declines?",
    "What are the main ideas of the London System (1.d4 2.Nf3 3.Bf4)? Why is it popular at club level?",
    "Explain the King's Gambit (1.e4 e5 2.f4). Is it sound at high level play? What are the risks and rewards?",
    "Describe the English Opening (1.c4). What type of positions does it lead to and what are White's plans?",
    "Explain the Scotch Game (1.e4 e5 2.Nf3 Nc6 3.d4). How does it differ strategically from the Italian Game?",
    "What is the Vienna Game (1.e4 e5 2.Nc3)? What are White's typical plans and piece placement?",
    "Explain the Catalan Opening (1.d4 Nf6 2.c4 e6 3.g3). Why is it a favorite of world champions?",
    "What is the Réti Opening (1.Nf3)? Explain its hypermodern philosophy.",
    "Explain the Four Knights Game (1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6). Is it drawish or can White fight for an advantage?",
    "What is the Bird's Opening (1.f4)? What are its strengths and weaknesses?",
    "Explain the Giuoco Piano vs the Evans Gambit. How does 4.b4 change the nature of the Italian Game?",
    "What is the Trompowsky Attack (1.d4 Nf6 2.Bg5)? When is it a good surprise weapon?",
    "Explain the Ponziani Opening (1.e4 e5 2.Nf3 Nc6 3.c3). What is White's idea with this modest move?",
    # Défenses noirs
    "Explain the Sicilian Defense (1.e4 c5). Why is it Black's most popular response to 1.e4?",
    "What is the French Defense (1.e4 e6)? Analyze its pawn structure, strengths, and weaknesses.",
    "Explain the Caro-Kann Defense (1.e4 c6). Why is it considered one of the most solid defenses?",
    "What is the Pirc Defense (1.e4 d6 2.d4 Nf6 3.Nc3 g6)? What type of player should choose it?",
    "Explain the Scandinavian Defense (1.e4 d5). What are the pros and cons of early queen development?",
    "What is the King's Indian Defense (1.d4 Nf6 2.c4 g6 3.Nc3 Bg7)? Explain the typical kingside attack.",
    "Explain the Nimzo-Indian Defense (1.d4 Nf6 2.c4 e6 3.Nc3 Bb4). Why do top GMs love this defense?",
    "What is the Slav Defense (1.d4 d5 2.c4 c6)? How does it compare to the Queen's Gambit Declined?",
    "Explain the Dutch Defense (1.d4 f5). What are its aggressive ideas and structural weaknesses?",
    "What is the Grünfeld Defense (1.d4 Nf6 2.c4 g6 3.Nc3 d5)? Explain how Black fights for the center.",
    "Explain the Alekhine Defense (1.e4 Nf6). What is the provocative idea behind this opening?",
    "What is the Benoni Defense (1.d4 Nf6 2.c4 c5)? Explain the pawn structure and typical plans.",
    "Explain the Petroff Defense (1.e4 e5 2.Nf3 Nf6). Why is it known as the most solid reply to 1.e4 e5?",
    "What is the Philidor Defense (1.e4 e5 2.Nf3 d6)? Is it passive or does Black have active plans?",
    "Explain the Owen Defense (1.e4 b6). What are the hypermodern ideas behind it?",
    # === ANALYSE DE POSITIONS (30) ===
    "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 (Sicilian Najdorf), what are White's main options and plans?",
    "After 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 O-O (Closed Ruy Lopez), what should White play and why?",
    "In the position after 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 Nbd7, explain White's plan with the minority attack.",
    "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 g6 (Sicilian Dragon), why does White play 6.Be3 followed by f3, Qd2, and O-O-O?",
    "Analyze the position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5. Why is this attack on f7 dangerous and how should Black defend?",
    "After 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O Nc6, explain the typical plans for both sides in the King's Indian.",
    "In the Sicilian Scheveningen (after 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e6), explain the Keres Attack with 6.g4.",
    "After 1.e4 e6 2.d4 d5 3.Nc3 Bb4 (French Winawer), analyze 4.e5. What are the consequences for the pawn structure?",
    "Analyze the Exchange Variation of the Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6. Why does White exchange and what are the resulting plans?",
    "After 1.d4 d5 2.c4 dxc4 (Queen's Gambit Accepted), explain why Black cannot hold the extra pawn and what White's plans are.",
    "In the Sveshnikov Sicilian after 1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e5 6.Ndb5 d6, explain why d5 is weak but Black is still OK.",
    "Analyze the position after 1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Nxd4 Bc5 (Scotch Game). What are the critical continuations?",
    "After 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.Qc2 (Nimzo-Indian Classical), why does White play Qc2 and what does Black do?",
    "Explain the typical middlegame after 1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 (Caro-Kann Classical). What are the plans for both sides?",
    "After 1.e4 e5 2.Nf3 Nf6 3.Nxe5 d6 4.Nf3 Nxe4 (Petroff), analyze the position. Why is it considered equal?",
    "In the Benko Gambit (1.d4 Nf6 2.c4 c5 3.d5 b5), explain why Black sacrifices a pawn and what long-term compensation Black gets.",
    "Analyze the Fried Liver Attack: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5 5.exd5 Nxd5 6.Nxf7. Is it sound?",
    "After 1.e4 c5 2.c3 (Alapin Sicilian), explain White's plan. How should Black respond?",
    "In the English Attack against the Najdorf (6.Be3 followed by f3, Qd2, g4), explain the strategic battle between the two sides.",
    "Analyze the Marshall Attack: 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 O-O 8.c3 d5. Why does Black sacrifice a pawn?",
    "After 1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6 5.e3 Nbd7 6.Bd3 dxc4 7.Bxc4 b5 (Semi-Slav Meran), explain the sharp play that follows.",
    "In the Advance French (1.e4 e6 2.d4 d5 3.e5), explain Black's typical plan with ...c5 and piece pressure on d4.",
    "Analyze the Smith-Morra Gambit (1.e4 c5 2.d4 cxd4 3.c3). Is the pawn sacrifice justified? What compensation does White get?",
    "After 1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.cxd5 Nxd5 5.e4 Nxc3 6.bxc3 Bg7 (Grünfeld Exchange), explain why White's big center might be a target.",
    "In the Tarrasch Defense (1.d4 d5 2.c4 e6 3.Nc3 c5), explain the isolated queen pawn and whether it is a strength or weakness.",
    "Analyze the Grand Prix Attack against the Sicilian (1.e4 c5 2.Nc3 Nc6 3.f4). What are White's attacking ideas?",
    "After 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5 (Scandinavian), explain Black's plan to develop while keeping the queen active.",
    "In the Closed Sicilian (1.e4 c5 2.Nc3 Nc6 3.g3), explain how this differs from the Open Sicilian and what type of game results.",
    "Analyze the position after 1.e4 e5 2.f4 exf4 3.Nf3 g5 (King's Gambit Accepted). What are White's attacking chances?",
    "After 1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 Be7 5.Nf3 O-O 6.O-O dxc4 (Open Catalan), explain how White recovers the pawn and maintains pressure.",
    # === COMPARAISONS ET STRATÉGIE (20) ===
    "Compare the Sicilian Najdorf and the Sicilian Dragon. Which is more aggressive and which is more solid?",
    "Compare the French Defense and the Caro-Kann. Both start with ...e6 or ...c6 against 1.e4 — what are the key differences?",
    "What are the differences between the Queen's Gambit Declined and the Slav Defense? Which gives Black more active play?",
    "Compare the King's Indian Defense and the Grünfeld Defense. Both involve ...g6 and ...Bg7 — how do the strategies differ?",
    "Italian Game vs Ruy Lopez: both develop the bishop early. What are the strategic differences between Bc4 and Bb5?",
    "Compare open games (1.e4 e5) vs semi-open games (1.e4 c5/e6/c6). What types of skills does each require?",
    "Compare 1.e4 and 1.d4 as first moves. What styles of play do they lead to?",
    "What is the difference between classical and hypermodern opening philosophy? Give examples of each.",
    "Compare the Nimzo-Indian and the Queen's Indian Defense. When should Black choose one over the other?",
    "Explain the concept of a gambit in chess openings. Compare the Queen's Gambit, King's Gambit, and Benko Gambit.",
    "What is the difference between playing for an attack and playing for positional advantage in the opening?",
    "Compare Anti-Sicilian systems (Alapin, Smith-Morra, Grand Prix) to the Open Sicilian. When should White avoid 2.Nf3 3.d4?",
    "Explain the concept of the pawn center in openings. Compare a broad center (e4+d4) vs a flexible center (c4+Nf3).",
    "Compare the Exchange variations in different openings (Exchange French, Exchange Slav, Exchange Ruy Lopez). Are they drawish?",
    "What are the pros and cons of early castling vs delayed castling in various openings?",
    "Compare the fianchetto bishop (g2 or g7) vs the classical bishop development (c4/f4 or c5/f5). When is each better?",
    "Explain how pawn structure in the opening determines middlegame and endgame plans. Give 3 examples.",
    "What is the difference between a sound gambit and an unsound gambit? Give examples from common openings.",
    "Compare rapid development openings (Italian, Scotch) vs slow strategic openings (English, Réti). What determines the choice?",
    "Explain the concept of transpositions in chess openings. Give 3 examples where different move orders reach the same position.",
    # === PÉDAGOGIE ET RECOMMANDATIONS (20) ===
    "What are the best openings for a complete beginner (under 1000 ELO) to learn? Explain your reasoning.",
    "What opening repertoire would you recommend for an intermediate player (1200-1500 ELO) playing White?",
    "What opening repertoire would you recommend for an intermediate player (1200-1500 ELO) playing Black against 1.e4?",
    "What opening repertoire would you recommend for an intermediate player (1200-1500 ELO) playing Black against 1.d4?",
    "Explain the concept of development and tempo in chess openings. Give examples of traps caused by poor development.",
    "What are the most common mistakes beginners make in the opening? List 5 with explanations.",
    "Explain the opening principles: control the center, develop pieces, castle early, connect rooks. Why does each matter?",
    "What is an opening trap? Describe 3 famous opening traps and how to avoid them.",
    "How should you study chess openings effectively? What is more important: memorizing moves or understanding ideas?",
    "Explain what happens when you ignore opening principles. Analyze the Scholar's Mate attempt (1.e4 e5 2.Qh5) and why it fails against correct play.",
    "What openings should you avoid as a beginner and why? Give 3 examples with explanations.",
    "How do you choose an opening that fits your playing style? Compare aggressive vs positional player preferences.",
    "Explain the importance of knowing the first 5-10 moves of your openings well. What happens when you leave theory?",
    "What is preparation in chess? How do strong players prepare their openings for specific opponents?",
    "Explain why the move order matters in openings. Give an example where playing moves in the wrong order loses.",
    "What are the most important endgame concepts that come directly from opening choices? Give 3 examples.",
    "How has computer analysis changed opening theory? Give examples of openings that were re-evaluated.",
    "Explain the concept of novelty in chess openings. Why do top players search for new moves?",
    "What is the best way to build an opening repertoire from scratch? Outline a step-by-step approach.",
    "Explain why understanding typical middlegame plans is more important than memorizing opening moves. Give examples.",
]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def generate_one(instruction, temperature, index, total):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction}
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                logprobs=True,
                top_logprobs=1
            )

            content = response.choices[0].message.content
            logprobs = response.choices[0].logprobs

            if content and len(content) > 100:
                tokens_used = response.usage.total_tokens
                print(f"  [{index+1}/{total}] OK ({tokens_used} tokens)")
                return {
                    "instruction": instruction,
                    "output": content,
                    "logprobs_data": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in logprobs.content
                    ] if logprobs and logprobs.content else [],
                    "temperature": temperature,
                    "stage": f"stage{'1' if temperature < 0.5 else '2'}"
                }
            else:
                print(f"  [{index+1}/{total}] Réponse trop courte, retry {attempt+1}")
        except Exception as e:
            print(f"  [{index+1}/{total}] Erreur: {e}, retry {attempt+1}")
            time.sleep(3)

    print(f"  [{index+1}/{total}] ÉCHEC")
    return None


def generate_stage(instructions, temperature, output_file):
    stage_name = "Stage 1 (τ=0.3)" if temperature < 0.5 else "Stage 2 (τ=0.9)"
    print(f"\n{'='*60}")
    print(f"  {stage_name} — {len(instructions)} questions, {MAX_WORKERS} workers")
    print(f"{'='*60}\n")

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_one, inst, temperature, i, len(instructions)): i
            for i, inst in enumerate(instructions)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: instructions.index(x["instruction"]))

    elapsed = time.time() - start
    print(f"\n{stage_name} terminé : {len(results)}/{len(instructions)} OK en {elapsed:.0f}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Sauvegardé dans {output_file}")

    return results


if __name__ == "__main__":
    print(f"Questions : {len(chess_instructions)}")
    print(f"Modèle : {TEACHER_MODEL}")
    print(f"Workers parallèles : {MAX_WORKERS}")

    stage1_data = generate_stage(chess_instructions, temperature=0.3, output_file="stage1_raw.json")
    stage2_data = generate_stage(chess_instructions, temperature=0.9, output_file="stage2_raw.json")

    print(f"\n{'='*60}")
    print(f"  TERMINÉ")
    print(f"  Stage 1 : {len(stage1_data)} exemples")
    print(f"  Stage 2 : {len(stage2_data)} exemples")
    print(f"{'='*60}")
