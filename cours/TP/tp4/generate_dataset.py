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
    # =====================================================================
    # 1. OPENING THEORY — WHITE SYSTEMS (40)
    # =====================================================================
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
    "What is the Colle System (1.d4 d5 2.Nf3 Nf6 3.e3)? How does it compare to the London System?",
    "Explain the Torre Attack (1.d4 Nf6 2.Nf3 e6 3.Bg5). What are White's plans against the King's Indian setup?",
    "What is the Veresov Opening (1.d4 d5 2.Nc3 Nf6 3.Bg5)? Why is it considered offbeat?",
    "Explain the King's Indian Attack (1.Nf3 d5 2.g3). How does White build a flexible setup?",
    "What is the Stonewall Attack (1.d4 2.e3 3.f4 4.Bd3)? Explain its fixed pawn structure and plans.",
    "Describe the Larsen Opening (1.b3). What are the strategic ideas behind this unusual first move?",
    "Explain the Nimzo-Larsen Attack (1.b3) compared to 1.Nf3. When is each preferable?",
    "What is the Grob Opening (1.g4)? Is it playable or just a surprise weapon?",
    "Explain the Alapin Sicilian (1.e4 c5 2.c3) from White's perspective. What are the main plans and typical pawn structures?",
    "What is the Grand Prix Attack (1.e4 c5 2.Nc3 Nc6 3.f4) and why do club players like it against the Sicilian?",
    "Explain the Closed Sicilian (1.e4 c5 2.Nc3 followed by g3 and Bg2). What kind of game does White want?",
    "What is the Smith-Morra Gambit (1.e4 c5 2.d4 cxd4 3.c3)? Explain why White sacrifices a pawn.",
    "Describe the Maroczy Bind structure (pawns on c4 and e4 against the Sicilian). How does White use this setup?",
    "What is the Exchange Variation of the Ruy Lopez (4.Bxc6)? Why did Fischer play it?",
    "Explain the Worrall Attack in the Ruy Lopez (6.Qe2 instead of 6.Re1). What is the strategic idea?",
    "What is the Anti-Berlin in the Ruy Lopez? Why do many players avoid the Berlin Wall at club level?",
    "Explain the Italian Game's Giuoco Pianissimo (slow Italian). How does White build pressure gradually?",
    "What is the Max Lange Attack (1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.d4 exd4 5.O-O)? Analyze the tactical complications.",
    "Explain the Fried Liver Attack setup. At what level is it effective and how should White prepare it?",
    "What is the Yugoslav Attack against the Sicilian Dragon? Describe White's plan step by step.",
    "Explain the Richter-Rauzer Attack (1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 Nc6 6.Bg5). What are White's ideas?",
    "What is the English Attack in the Najdorf (6.Be3, f3, Qd2, g4, O-O-O)? Describe the attacking scheme.",
    "Explain the Bayonet Attack in the King's Indian (1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O Nc6 8.d5 Ne7 9.b4). What is White trying to achieve?",
    "What is the Samisch variation of the King's Indian (with f3)? How does White plan to attack?",
    "Explain the Botvinnik System in the English (1.c4 e5 2.Nc3 Nc6 3.g3 g6 4.Bg2 Bg7 5.e4). What kind of center does White build?",

    # =====================================================================
    # 2. OPENING THEORY — BLACK DEFENSES (40)
    # =====================================================================
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
    "What is the Modern Defense (1.e4 g6)? How does it differ from the Pirc Defense?",
    "Explain the Budapest Gambit (1.d4 Nf6 2.c4 e5). What traps can Black set and is the gambit sound?",
    "What is the Bogo-Indian Defense (1.d4 Nf6 2.c4 e6 3.Nf3 Bb4+)? How does it differ from the Nimzo-Indian?",
    "Explain the Queen's Indian Defense (1.d4 Nf6 2.c4 e6 3.Nf3 b6). What is Black's strategic concept?",
    "What is the Tarrasch Defense (1.d4 d5 2.c4 e6 3.Nc3 c5)? Explain the isolated queen pawn debate.",
    "Explain the Semi-Slav Defense (1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6). Why is it one of Black's richest systems?",
    "What is the Chigorin Defense (1.d4 d5 2.c4 Nc6)? Why is it unconventional and what are its ideas?",
    "Explain the Sicilian Najdorf (5...a6). Why has it been the choice of Fischer, Kasparov, and many champions?",
    "What is the Sicilian Dragon? Explain the fianchetto concept and Black's counterplay on the c-file.",
    "Explain the Sicilian Sveshnikov (1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e5). Why accept the weak d5 square?",
    "What is the Sicilian Scheveningen (with ...e6 and ...d6)? What are Black's typical piece placements?",
    "Explain the Sicilian Taimanov (1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 Nc6). What flexibility does this offer Black?",
    "What is the Sicilian Kan (1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 a6)? How does it differ from the Taimanov?",
    "Explain the Accelerated Dragon (1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 g6). Why skip ...d6?",
    "What is the Kalashnikov Sicilian (5...e5 with Nc6 instead of Nf6)? How does it compare to the Sveshnikov?",
    "Explain the French Advance Variation (3.e5). What are Black's plans against the pawn chain?",
    "What is the French Tarrasch Variation (3.Nd2)? Why does White play Nd2 instead of Nc3?",
    "Explain the French Classical (3.Nc3 Nf6). What are the main lines and ideas for both sides?",
    "What is the Caro-Kann Advance Variation (3.e5)? How does Black generate counterplay?",
    "Explain the Caro-Kann Classical (3.Nc3 dxe4 4.Nxe4 Bf5). Why is Bf5 such an important move?",
    "What is the Exchange French (3.exd5 exd5)? Is it really as drawish as its reputation?",
    "Explain the Stonewall Dutch (1.d4 f5 with ...e6, ...d5, ...c6). What are its strengths on the kingside?",
    "What is the Leningrad Dutch (with ...g6 and ...Bg7)? How does it combine Dutch and King's Indian ideas?",
    "Explain the Classical Dutch (with ...e6, ...Be7). How does Black plan the middlegame?",
    "What is the Berlin Defense in the Ruy Lopez (3...Nf6)? Why did it become dominant after Kramnik-Kasparov 2000?",

    # =====================================================================
    # 3. SPECIFIC VARIATIONS & SUB-LINES (80)
    # =====================================================================
    "Explain the Najdorf Poisoned Pawn variation (6.Bg5 e6 7.f4 Qb6). Why is taking on b2 so risky yet playable?",
    "What is the Najdorf 6.Be2 line? How does it differ in character from 6.Bg5 and 6.Be3?",
    "Explain the Sicilian Dragon Yugoslav Attack (9.Bc4). What is the typical race between White's kingside and Black's queenside attack?",
    "What is the Classical Sicilian (1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 Nc6)? How does it compare to the Najdorf?",
    "Explain the Rossolimo Sicilian (1.e4 c5 2.Nf3 Nc6 3.Bb5). Why is it a popular Anti-Sicilian?",
    "What is the Moscow Variation of the Sicilian (1.e4 c5 2.Nf3 d6 3.Bb5+)? What does White achieve?",
    "Explain the Breyer Variation of the Ruy Lopez (9...Nb8). Why does Black retreat the knight?",
    "What is the Chigorin Variation of the Ruy Lopez (9...Na5)? What are Black's plans?",
    "Explain the Open Variation of the Ruy Lopez (5...Nxe4). How does Black justify giving up the center?",
    "What is the Zaitsev Variation of the Ruy Lopez (9...Bb7)? Why is it considered one of Black's best systems?",
    "Explain the Arkhangelsk Variation of the Ruy Lopez (5...b5 6.Bb3 Bb7). What is the modern approach?",
    "What is the QGD Orthodox Defense (with ...Be7, ...O-O, ...Nbd7)? Explain White's typical plan.",
    "Explain the QGD Ragozin Variation (3.Nc3 Nf6 4.Nf3 Bb4). How does it mix Nimzo-Indian ideas with QGD?",
    "What is the QGD Vienna Variation (4.Nf3 dxc4)? Why does Black capture early?",
    "Explain the QGD Lasker Defense (with ...Ne4). What simplification does Black achieve?",
    "What is the Anti-Moscow Gambit in the Semi-Slav (5.Bg5 h6 6.Bh4)? Explain the sharp complications.",
    "Explain the Botvinnik Variation of the Semi-Slav (5.Bg5 dxc4 6.e4). Why is it one of the sharpest openings?",
    "What is the Meran Variation of the Semi-Slav? Explain the typical pawn advances on the queenside.",
    "Explain the King's Indian Classical Variation (with Nf3, Be2, O-O). What is the standard plan for both sides?",
    "What is the King's Indian Sämisch (with f3)? How does it change the character compared to the Classical?",
    "Explain the King's Indian Four Pawns Attack (f4 on move 5). Is it dangerous for Black or overambitious?",
    "What is the King's Indian Averbakh Variation (Bg5)? How does White try to prevent ...f5?",
    "Explain the Grünfeld Exchange Variation (4.cxd5 Nxd5 5.e4). How does Black attack the center?",
    "What is the Grünfeld Russian System (with Nf3 and Rb1)? How does it avoid the main Exchange lines?",
    "Explain the Nimzo-Indian Rubinstein Variation (4.e3). What are the typical pawn structures?",
    "What is the Nimzo-Indian Sämisch Variation (4.a3)? Why does White force the bishop exchange?",
    "Explain the Nimzo-Indian Leningrad Variation (4.Bg5). What pressure does White create?",
    "What is the Nimzo-Indian Classical (4.Qc2)? Why is preventing doubled pawns so important?",
    "Explain the Catalan Open Variation (Black takes on c4). How does White's Bg2 help recover the pawn?",
    "What is the Catalan Closed Variation (Black keeps the center closed)? What positional pressure does White have?",
    "Explain the English Symmetrical Variation (1.c4 c5). What are the typical pawn structures?",
    "What is the English Four Knights (1.c4 e5 2.Nc3 Nf6 3.Nf3 Nc6)? How does it differ from 1.e4 versions?",
    "Explain the English Reversed Sicilian concept. Why can White benefit from the extra tempo?",
    "What is the Two Knights Defense (1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6)? How does Black counter-attack the center?",
    "Explain the Traxler Counter-Attack (4.Ng5 Bc5). Is it sound or just a trap?",
    "What is the Halloween Gambit (1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6 4.Nxe5)? Analyze its soundness.",
    "Explain the Scotch Gambit (1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Bc4). How does it combine Italian and Scotch ideas?",
    "What is the Belgrade Gambit (Four Knights with d4)? Is it a real gambit or does White equalize?",
    "Explain the Philidor Hanham Variation (...Nbd7, ...Be7). Is it a solid modern treatment?",
    "What is the Latvian Gambit (1.e4 e5 2.Nf3 f5)? Why is it considered dubious but tricky?",
    "Explain the Elephant Gambit (1.e4 e5 2.Nf3 d5). Can Black justify this early center strike?",
    "What is the Damiano Defense (1.e4 e5 2.Nf3 f6)? Why is it considered one of the worst defenses?",
    "Explain the Center Game (1.e4 e5 2.d4 exd4 3.Qxd4). Why is early queen development bad here for White?",
    "What is the Danish Gambit (1.e4 e5 2.d4 exd4 3.c3)? How much material does White sacrifice and is it worth it?",
    "Explain the King's Gambit Declined (1.e4 e5 2.f4 Bc5). Why is this considered Black's most solid response?",
    "What is the Falkbeer Counter-Gambit (1.e4 e5 2.f4 d5)? How does Black seize the initiative?",
    "Explain the Sicilian O'Kelly Variation (2...a6). What is Black's idea with this early pawn move?",
    "What is the Wing Gambit against the Sicilian (1.e4 c5 2.b4)? Is it playable at competitive level?",
    "Explain the French Rubinstein Variation (3.Nc3 dxe4 4.Nxe4). What type of position results?",
    "What is the French Burn Variation (3.Nc3 Nf6 4.Bg5 dxe4)? How does Black simplify?",
    "Explain the Caro-Kann Two Knights (3.Nc3 dxe4 4.Nxe4 Nf6 5.Nxf6+ exf6). What does Black get for the damaged structure?",
    "What is the Caro-Kann Panov-Botvinnik Attack (3.exd5 cxd5 4.c4)? How does it create an IQP position?",
    "Explain the Caro-Kann Fantasy Variation (3.f3). What aggressive ideas does White have?",
    "What is the Scandinavian Icelandic-Palme Gambit (3...Qd8 4.d4 Nf6 5.Nf3 Bg4)? Is it a real gambit?",
    "Explain the Scandinavian Portuguese Variation (3...Qd6). What are the ideas compared to 3...Qa5?",
    "What is the Alekhine Four Pawns Attack (1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.f4)? Is it dangerous for Black?",
    "Explain the Alekhine Modern Variation (3...d6 4.Nf3). Why is this White's most common continuation?",
    "What is the Dutch Staunton Gambit (1.d4 f5 2.e4)? How should Black respond?",
    "Explain the Benko Gambit Accepted (4.cxb5 a6). What compensation does Black have on the queenside?",
    "What is the Benko Gambit Declined (4.Nf3 or 4.a4)? How does White try to keep the extra pawn?",
    "Explain the Czech Benoni (1.d4 Nf6 2.c4 c5 3.d5 e5). What is the closed pawn structure and plans?",
    "What is the Modern Benoni (3...e6 4.Nc3 exd5 5.cxd5 d6)? Explain the typical pawn structure imbalance.",
    "Explain the Blumenfeld Gambit (1.d4 Nf6 2.c4 e6 3.Nf3 c5 4.d5 b5). What does Black achieve?",
    "What is the Old Indian Defense (1.d4 Nf6 2.c4 d6)? How does it differ from the King's Indian?",
    "Explain the London System reversed — how should Black respond to 1.d4 2.Bf4 with ...d5 setups?",
    "What is the Torre Attack reversed — what are Black's best plans against Bg5 systems?",
    "Explain the Symmetrical English (1.c4 c5 2.Nc3 Nc6 3.g3 g6). What are the typical hedgehog and Maroczy structures?",
    "What is the Réti Accepted (1.Nf3 d5 2.c4 dxc4)? Can Black hold the pawn?",
    "Explain the King's Fianchetto against the English (1.c4 e5 2.g3). What does Black achieve with an early ...e5?",
    "What is the Polish Opening (1.b4)? What are the strategic ideas and is it playable seriously?",
    "Explain the Benko Opening / Hungarian Opening (1.g3). How is it related to the King's Indian Attack?",
    "What is the Van Geet Opening (1.Nc3)? What flexibility does it offer White?",
    "Explain the St. George Defense (1.e4 a6). What is Black's provocative idea?",
    "What is the Hippopotamus Defense? Explain how Black sets up with ...b6, ...g6, ...d6, ...e6.",
    "Explain the Czech Defense / Pribyl Defense (1.e4 d6 2.d4 Nf6 3.Nc3 c6). How does it blend Pirc and Caro-Kann?",
    "What is the Lion Defense (a Philidor-like setup with ...Nbd7 and ...Be7)? Is it viable at club level?",
    "Explain the Hedgehog formation from Black's side. What are the typical piece placements and pawn breaks?",

    # =====================================================================
    # 4. POSITION ANALYSIS & CRITICAL MOMENTS (80)
    # =====================================================================
    "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 (Sicilian Najdorf), what are White's main options and plans?",
    "After 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 O-O (Closed Ruy Lopez), what should White play and why?",
    "In the position after 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 Nbd7, explain White's plan with the minority attack.",
    "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 g6 (Sicilian Dragon), why does White play 6.Be3 followed by f3, Qd2, and O-O-O?",
    "Analyze the position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5. Why is this attack on f7 dangerous and how should Black defend?",
    "After 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O Nc6, explain the typical plans for both sides in the King's Indian.",
    "In the Sicilian Scheveningen (after 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e6), explain the Keres Attack with 6.g4.",
    "After 1.e4 e6 2.d4 d5 3.Nc3 Bb4 (French Winawer), analyze 4.e5. What are the consequences for the pawn structure?",
    "Analyze the Exchange Variation of the Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6. Why does White exchange?",
    "After 1.d4 d5 2.c4 dxc4 (Queen's Gambit Accepted), explain why Black cannot hold the extra pawn.",
    "In the Sveshnikov Sicilian after 5...e5 6.Ndb5 d6, explain why d5 is weak but Black is still OK.",
    "Analyze the position after 1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Nxd4 Bc5 (Scotch Game). What are the critical continuations?",
    "After 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.Qc2 (Nimzo-Indian Classical), why does White play Qc2?",
    "Explain the typical middlegame after 1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 (Caro-Kann Classical).",
    "After 1.e4 e5 2.Nf3 Nf6 3.Nxe5 d6 4.Nf3 Nxe4 (Petroff), analyze why this position is considered equal.",
    "In the Benko Gambit (1.d4 Nf6 2.c4 c5 3.d5 b5), explain Black's long-term compensation.",
    "Analyze the Fried Liver Attack: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5 5.exd5 Nxd5 6.Nxf7. Is it sound?",
    "After 1.e4 c5 2.c3 (Alapin Sicilian), explain White's plan. How should Black respond?",
    "In the English Attack against the Najdorf (6.Be3 followed by f3, Qd2, g4), explain the strategic battle.",
    "Analyze the Marshall Attack: 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 O-O 8.c3 d5. Why does Black sacrifice a pawn?",
    "After 1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6 5.e3 Nbd7 6.Bd3 dxc4 7.Bxc4 b5 (Semi-Slav Meran), explain the sharp play.",
    "In the Advance French (1.e4 e6 2.d4 d5 3.e5), explain Black's typical plan with ...c5.",
    "Analyze the Smith-Morra Gambit (1.e4 c5 2.d4 cxd4 3.c3). What compensation does White get?",
    "After 1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.cxd5 Nxd5 5.e4 Nxc3 6.bxc3 Bg7 (Grünfeld Exchange), why is White's center a target?",
    "In the Tarrasch Defense (1.d4 d5 2.c4 e6 3.Nc3 c5), explain whether the isolated queen pawn is a strength or weakness.",
    "Analyze the Grand Prix Attack against the Sicilian (1.e4 c5 2.Nc3 Nc6 3.f4). What are White's attacking ideas?",
    "After 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5, explain Black's plan in the Scandinavian.",
    "In the Closed Sicilian (1.e4 c5 2.Nc3 Nc6 3.g3), explain how this differs from the Open Sicilian.",
    "Analyze the position after 1.e4 e5 2.f4 exf4 3.Nf3 g5 (King's Gambit Accepted). What are White's attacking chances?",
    "After 1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 Be7 5.Nf3 O-O 6.O-O dxc4 (Open Catalan), explain how White recovers the pawn.",
    "Analyze the critical moment in the Najdorf after 6.Bg5 e6 7.f4 Be7 8.Qf3 Qc7. What are the plans for both sides?",
    "In the King's Indian, after 7.O-O Nc6 8.d5 Ne7, explain the pawn break ...f5 and when to play it.",
    "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Be3 e5 7.Nb3, explain why this retreat is important.",
    "Analyze the position after 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.e3 O-O 5.Bd3 d5 6.Nf3 c5. What pawn structure results?",
    "In the French Defense after 3.Nd2 c5 4.exd5 exd5, explain the IQP position and plans for both sides.",
    "After the Caro-Kann Advance 3.e5 Bf5 4.Nf3 e6 5.Be2 c5, analyze the pawn tension and plans.",
    "In the Ruy Lopez after 9.h3 (preventing Bg4), what are Black's main plans: ...Nb8-d7, ...Bb7, or ...Na5?",
    "Analyze the critical position in the Sicilian Dragon after White plays h4-h5. How should Black react?",
    "After 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4 6.cxd4 Bb4+, analyze this Italian Game line.",
    "In the Grünfeld after 5.e4 Nxc3 6.bxc3 Bg7 7.Nf3 c5 8.Be2 O-O 9.O-O, why does Black play ...cxd4 and ...Nc6?",
    "Analyze the position in the QGD after 7.Rc1 c6 8.Bd3 dxc4 9.Bxc4 Nd5. What is Black's plan?",
    "After 1.e4 e5 2.Nf3 Nc6 3.Bb5 f5 (Schliemann/Jaenisch), analyze this gambit. Is it sound?",
    "In the Slav after 3.Nf3 Nf6 4.Nc3 dxc4 5.a4 Bf5, explain the typical position and plans.",
    "After 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.cxd5 exd5 (Exchange QGD), explain why this is considered boring and what each side should aim for.",
    "Analyze the position in the Benoni after 5.cxd5 d6 6.e4 g6 7.Nf3 Bg7 8.Be2 O-O. Explain the light square strategy.",
    "In the Open Ruy Lopez after 5...Nxe4 6.d4 b5 7.Bb3 d5 8.dxe5 Be6, analyze the complications.",
    "After 1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 Nc6 5.Nc3 Qc7 (Taimanov), explain both sides' plans.",
    "Analyze the typical King's Indian middlegame after Black plays ...f5 and White responds with exf5. What are the consequences?",
    "In the Catalan after 1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 dxc4 5.Nf3 a6, analyze White's pressure on the long diagonal.",
    "After 1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Nxd4 Nf6 5.Nxc6 bxc6 (Scotch), explain the resulting pawn structure.",

    # =====================================================================
    # 5. STRATEGY & COMPARISONS (60)
    # =====================================================================
    "Compare the Sicilian Najdorf and the Sicilian Dragon. Which is more aggressive and which is more solid?",
    "Compare the French Defense and the Caro-Kann. Both play against 1.e4 — what are the key differences?",
    "What are the differences between the Queen's Gambit Declined and the Slav Defense?",
    "Compare the King's Indian Defense and the Grünfeld Defense. How do the strategies differ?",
    "Italian Game vs Ruy Lopez: what are the strategic differences between Bc4 and Bb5?",
    "Compare open games (1.e4 e5) vs semi-open games (1.e4 c5/e6/c6). What skills does each require?",
    "Compare 1.e4 and 1.d4 as first moves. What styles of play do they lead to?",
    "What is the difference between classical and hypermodern opening philosophy? Give examples.",
    "Compare the Nimzo-Indian and the Queen's Indian Defense. When should Black choose one over the other?",
    "Explain the concept of a gambit in chess openings. Compare the Queen's Gambit, King's Gambit, and Benko Gambit.",
    "What is the difference between playing for an attack and playing for positional advantage in the opening?",
    "Compare Anti-Sicilian systems (Alapin, Smith-Morra, Grand Prix) to the Open Sicilian.",
    "Explain the concept of the pawn center in openings. Compare a broad center vs a flexible center.",
    "Compare the Exchange variations in different openings. Are they drawish?",
    "What are the pros and cons of early castling vs delayed castling in various openings?",
    "Compare the fianchetto bishop vs the classical bishop development. When is each better?",
    "Explain how pawn structure in the opening determines middlegame and endgame plans. Give 3 examples.",
    "What is the difference between a sound gambit and an unsound gambit? Give examples.",
    "Compare rapid development openings (Italian, Scotch) vs slow strategic openings (English, Réti).",
    "Explain the concept of transpositions in chess openings. Give 3 examples.",
    "Compare the Sicilian Scheveningen and the Sicilian Najdorf. How do the pawn structures differ?",
    "What is the strategic difference between playing ...d5 in one move vs preparing it with ...c6 first?",
    "Compare the Nimzo-Indian 4.e3 (Rubinstein) vs 4.Qc2 (Classical). What does each prioritize?",
    "Explain the trade-off between space advantage and pawn weakness. Give 3 opening examples.",
    "Compare kingside attacks in the Sicilian (White) vs the King's Indian (Black). What are the similarities?",
    "What is the difference between a closed center and a locked center? How do they affect opening strategy?",
    "Compare the QGD Carlsbad structure (after exchange on d5) with the IQP structure. Which is better for whom?",
    "Explain why some openings lead to sharp tactical play and others to quiet positional games. What determines this?",
    "Compare the Ruy Lopez Breyer (Nb8) vs Chigorin (Na5) systems. What is the philosophical difference?",
    "What is the role of the dark-squared bishop in 1.d4 openings? Compare positions with and without it.",
    "Explain the concept of a bad bishop in openings. In which openings does Black commonly end up with one?",
    "Compare the piece activity in the Italian Game vs the Scotch Game after 5 moves. Which gives more open lines?",
    "What is the difference between a strategic sacrifice and a tactical sacrifice in openings? Give examples.",
    "Compare the Sicilian Taimanov and the Sicilian Kan. What flexibility difference does ...Nc6 vs ...a6 provide?",
    "Explain the concept of overextension in chess openings. When does gaining space become a liability?",
    "Compare the Stonewall pawn structure (d4-e3-f4 or d5-e6-f5) for White and Black. Who benefits more?",
    "What role does king safety play in opening choice? Compare castling patterns in the Sicilian vs the QGD.",
    "Explain why the move ...c5 is thematic in so many Black defenses (French, QGD, Benoni, Sicilian).",
    "Compare the Maroczy Bind vs the Hedgehog structure. How do piece placements differ?",
    "What is the difference between a symmetrical and asymmetrical pawn structure? Which favors the attacker?",
    "Explain why doubled pawns can be both a strength and a weakness in openings. Give examples from the Nimzo-Indian.",
    "Compare the central tension in the French Defense vs the Caro-Kann. How does ...e6 vs ...c6 change the game?",
    "What is the concept of piece harmony in openings? Give an example of good and bad piece coordination.",
    "Compare the English Opening and the Réti Opening. Can they transpose into each other?",
    "Explain the concept of prophylaxis in openings. Give examples from the Ruy Lopez and the KID.",
    "Compare the King's Gambit and the Evans Gambit. Which offers better compensation and why?",
    "What is the role of the queen in the opening? Compare openings where the queen develops early vs stays back.",
    "Explain the difference between a temporary pawn sacrifice and a permanent one. Give opening examples.",
    "Compare how White fights for an advantage in 1.e4 e5 openings vs 1.d4 d5 openings. Different strategies?",
    "What makes an opening 'refuted' vs 'dubious' vs 'playable'? Give examples of each category.",
    "Compare the strategic themes when Black plays ...e5 against 1.d4 (Budapest, Old Benoni) vs ...c5 (Benoni).",
    "Explain why some openings are more popular at amateur level than GM level. Give 3 examples.",
    "Compare the light square and dark square strategies in the French Defense. Which bishop is stronger?",
    "What is the concept of a pawn chain? Analyze how to attack a pawn chain in the French and King's Indian.",
    "Explain the concept of central control vs central occupation. Which openings favor each approach?",
    "Compare the typical endgames arising from the Ruy Lopez Exchange vs the QGD Exchange. Which is more complex?",
    "What strategic themes are common to the Catalan and the Réti? How does the Bg2 bishop influence play?",
    "Explain the concept of a space advantage in chess. Which openings give White or Black more space?",
    "Compare the main ideas of e4-openings vs d4-openings from Black's perspective. Which is harder to face?",
    "Explain the concept of piece activity vs material in openings. When is a piece sacrifice justified?",

    # =====================================================================
    # 6. TACTICS, TRAPS & COMBINATIONS (50)
    # =====================================================================
    "What is an opening trap? Describe 3 famous opening traps and how to avoid them.",
    "Explain the Legal Trap (1.e4 e5 2.Nf3 d6 3.Bc4 Bg4 4.Nc3 g6 5.Nxe5). How does it work?",
    "Analyze the Noah's Ark Trap in the Ruy Lopez. How does Black win White's bishop?",
    "Explain the Siberian Trap in the Smith-Morra Gambit. What is the winning sequence for Black?",
    "What is the Lasker Trap in the Queen's Gambit Accepted? Show the tactical idea.",
    "Analyze the Elephant Trap in the QGD (5...Nxe4 after 5.Bg5). How does Black win material?",
    "Explain the Fishing Pole Trap in the Ruy Lopez (with ...h5-h4 and ...Ng4). What is the attacking idea?",
    "What is the Mortimer Trap in the Ruy Lopez (3...Nf6 4.Ba4 d6 5.d3 Bg4)? Show the tactic.",
    "Analyze the Blackburne Shilling Gambit (1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4). Is it a real trap?",
    "Explain the Englund Gambit traps (1.d4 e5). What can happen if White is not careful?",
    "What is the Scholar's Mate (1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#)? How to prevent it?",
    "Analyze the trap in the Budapest Gambit: 1.d4 Nf6 2.c4 e5 3.dxe5 Ng4 4.Bf4 Nc6 5.Nf3 Bb4+. What happens?",
    "Explain the Stafford Gambit traps (1.e4 e5 2.Nf3 Nf6 3.Nxe5 Nc6). What tricks does Black have?",
    "What is the Marshall Trap in the Italian Game? Show the key tactical idea.",
    "Analyze the typical Greek Gift sacrifice (Bxh7+) in openings. In which positions does it work?",
    "Explain the double bishop sacrifice (Lasker's combination). In which opening setups is it possible?",
    "What is the Muzio Gambit in the King's Gambit (sacrificing the knight on f7)? Is it sound?",
    "Analyze the typical f7 sacrifices in the Italian Game. When are they correct?",
    "Explain the concept of a discovered attack in opening tactics. Give 3 examples from common openings.",
    "What are the most common pins in the opening? Analyze Bg5 pinning Nf6, Bb5 pinning Nc6, etc.",
    "Explain the Boden's Mate pattern and in which openings it can arise.",
    "What is the concept of an opening novelty that is also a tactical trap? Give a historical example.",
    "Analyze the typical sacrifice on e6 in the Sicilian. When is Nxe6 or Bxe6 correct for White?",
    "Explain the Qa4+ trick in the Scandinavian. How can White exploit Black's queen position?",
    "What is the concept of a deflection in opening tactics? Give 2 examples from common openings.",
    "Analyze the typical exchange sacrifice (Rxc3 or Rxf3) in the Sicilian. When is it thematic for Black?",
    "Explain the Windmill tactic and in which opening positions it can occur.",
    "What opening traps exist in the London System? How can Black fall into trouble?",
    "Analyze the typical Nd5 sacrifice in the Sicilian. When is it correct and what does White get?",
    "Explain the concept of a poisoned pawn. Besides the Najdorf, where else does this theme appear?",
    "What tactical ideas exist in the King's Indian when Black plays ...f5 and the e4 pawn is attacked?",
    "Analyze the typical knight sacrifice on d5 in the French Defense. When is Nxd5 effective?",
    "Explain the typical Bxf7+ sacrifice in the Two Knights Defense. Why is it dangerous?",
    "What is the Fried Liver Attack (6.Nxf7) and the Lolli Attack (6.d4)? Compare the two approaches.",
    "Analyze common traps in the Petroff Defense. Can White punish Black's symmetrical play?",
    "Explain the typical central breakthrough (d4-d5 or e4-e5) as a tactical tool in openings.",
    "What traps exist in the Queen's Gambit Declined? Can Black fall into trouble with passive play?",
    "Analyze the Halosar Trap in the Blackmar-Diemer Gambit. How does White exploit development lead?",
    "Explain the concept of a zwischenzug (intermediate move) in openings. Give 2 examples.",
    "What is the concept of an interference in opening tactics? Give an example from a common line.",
    "Analyze typical back-rank threats that can arise from opening play. Which openings are prone to this?",
    "Explain the concept of a decoy sacrifice in openings. Give an example from the Sicilian or Italian.",
    "What tactical problems can arise from an uncastled king in the opening? Give 3 examples.",
    "Analyze the typical Nf5 sacrifice in the King's Indian or Sicilian. What does the attacking side get?",
    "Explain why developing the queen too early can lead to tactical problems. Give 3 opening examples.",
    "What is the concept of an overloaded piece in opening positions? Give an example.",
    "Analyze the Kieseritzky Gambit (1.e4 e5 2.f4 exf4 3.Nf3 g5 4.h4 g4 5.Ne5). What are White's tactical ideas?",
    "Explain the concept of a desperado (piece that is lost anyway making a capture). Give an opening example.",
    "What tactical themes arise from the Evan's Gambit? How does White exploit the open lines?",
    "Analyze the typical queen trap patterns in openings. In which openings can the queen get trapped?",

    # =====================================================================
    # 7. PAWN STRUCTURES & ENDGAME IMPLICATIONS (50)
    # =====================================================================
    "Explain the isolated queen pawn (IQP) structure. In which openings does it arise and what are the plans?",
    "What is the Carlsbad pawn structure (from the QGD Exchange)? Explain the minority attack and kingside plans.",
    "Explain the hanging pawns structure (c4+d4 for White or c5+d5 for Black). Strength or weakness?",
    "What is the Sicilian pawn structure (Black: d6+e7, White: e4)? How does it influence the middlegame?",
    "Explain the French pawn chain (White: d4-e5, Black: d5-e6). How should each side attack it?",
    "What is the Caro-Kann pawn structure after the exchange? How does it compare to the French?",
    "Explain the King's Indian pawn center (White: d4+c4, Black: d6+e5). What are the typical breaks?",
    "What is the Grünfeld center (White: c3+d4+e4 vs Black: none in center)? How does Black attack it?",
    "Explain the Benoni pawn structure (White: d5+e4, Black: c5+d6+e5). What are the asymmetric plans?",
    "What is the Stonewall pawn structure? Explain its role in the Dutch and as a White system.",
    "Explain the concept of a passed pawn created in the opening. In which structures does this happen?",
    "What is the Maroczy Bind pawn structure (c4+e4)? Why is it so restricting for Black?",
    "Explain the doubled pawns on c3 in the Nimzo-Indian. Are they a strength or weakness?",
    "What is the typical pawn structure after the Sicilian Exchange (exd5 cxd5)? Who benefits?",
    "Explain how the Ruy Lopez Exchange (Bxc6 dxc6) changes the pawn structure. What endgame advantages does White have?",
    "What pawn structure arises from the Scotch Game? How does it differ from the Italian?",
    "Explain the concept of a pawn majority in the opening. How does it relate to endgame prospects?",
    "What is the backward d-pawn in the Sicilian (after ...d6 and ...e5)? Is it a real weakness?",
    "Explain the pawn structure after the French Advance (e5). How does the fixed center affect both sides?",
    "What is the typical endgame that arises from the Berlin Ruy Lopez (3...Nf6 4.O-O Nxe4)? Why is it complex?",
    "Explain the concept of good knight vs bad bishop. In which opening pawn structures does this arise?",
    "What pawn structure arises from the Slav Exchange (exd5 cxd5 or cxd5 cxd5)? What are the plans?",
    "Explain the open e-file in the Petroff. How does it influence the middlegame and endgame?",
    "What is the pawn structure after the QGA (dxc4)? How does White's central majority play out?",
    "Explain the concept of pawn islands. Which opening structures lead to the fewest/most pawn islands?",
    "What pawn breaks are typical in the King's Indian (...f5 for Black, c5 for White)? When to play each?",
    "Explain the concept of a pawn lever. What are the main levers in the French, KID, and Benoni?",
    "What happens to the pawn structure after the typical Sicilian sacrifice ...Rxc3? How does it affect the endgame?",
    "Explain the doubled f-pawns structure (after Bxf6 gxf6). In which openings does this arise and is it good or bad?",
    "What is the hedgehog pawn structure (a6-b6-d6-e6)? How does Black use it to generate counterplay?",
    "Explain the concept of a pawn chain and how to attack its base. Use the French and KID as examples.",
    "What endgame themes arise from the Catalan Opening? How does the long diagonal influence endings?",
    "Explain the pawn structure after the Caro-Kann Panov-Botvinnik (c4xd5, d4). How is the IQP different here?",
    "What is the concept of central pawn tension? When should you maintain it vs resolve it?",
    "Explain how the choice of opening affects rook endgames. Compare open files in the Ruy Lopez vs the Sicilian.",
    "What pawn structure arises from 1.d4 d5 2.c4 e6 3.Nc3 c5 (Tarrasch)? Analyze the IQP implications.",
    "Explain the concept of a pawn storm. In which opening structures is it most effective?",
    "What is the Boleslavsky hole (d5 weakness in the Sicilian after ...e5)? Is it really a weakness?",
    "Explain the pawn structure similarities between the French Defense and the King's Indian (reversed).",
    "What endgame considerations should influence your choice between the Najdorf and the Dragon?",
    "Explain the concept of a mobile pawn center vs a fixed pawn center. Give 3 opening examples.",
    "What pawn structure arises in the English Opening Botvinnik System? How does the e4 pawn support plans?",
    "Explain the typical queenside pawn majority exploitation. Which openings give White this advantage?",
    "What is the concept of a pawn sacrifice for long-term structural advantage? Give 3 opening examples.",
    "Explain the rook's role behind passed pawns in the endgame. How does the opening pawn structure determine this?",
    "What is the typical bishop vs knight battle in the QGD? Which minor piece is better and why?",
    "Explain the concept of piece exchanges based on pawn structure. When should you trade pieces in the opening?",
    "What endgame techniques arise from the Exchange Slav (symmetrical pawns)? Is it always drawn?",
    "Explain the concept of the two bishops advantage. In which opening pawn structures is it most relevant?",
    "What is the concept of structural compensation for material? Give 3 opening examples.",

    # =====================================================================
    # 8. PEDAGOGY, RECOMMENDATIONS & STUDY (50)
    # =====================================================================
    "What are the best openings for a complete beginner (under 1000 ELO) to learn? Explain your reasoning.",
    "What opening repertoire would you recommend for an intermediate player (1200-1500 ELO) playing White?",
    "What opening repertoire would you recommend for an intermediate player (1200-1500 ELO) playing Black against 1.e4?",
    "What opening repertoire would you recommend for an intermediate player (1200-1500 ELO) playing Black against 1.d4?",
    "Explain the concept of development and tempo in chess openings. Give examples of traps caused by poor development.",
    "What are the most common mistakes beginners make in the opening? List 5 with explanations.",
    "Explain the opening principles: control the center, develop pieces, castle early, connect rooks.",
    "How should you study chess openings effectively? What is more important: memorizing moves or understanding ideas?",
    "Explain what happens when you ignore opening principles. Analyze the Scholar's Mate attempt.",
    "What openings should you avoid as a beginner and why? Give 3 examples.",
    "How do you choose an opening that fits your playing style? Compare aggressive vs positional preferences.",
    "Explain the importance of knowing the first 5-10 moves of your openings well.",
    "What is preparation in chess? How do strong players prepare their openings for specific opponents?",
    "Explain why the move order matters in openings. Give an example where wrong order loses.",
    "What are the most important endgame concepts that come directly from opening choices?",
    "How has computer analysis changed opening theory? Give examples of re-evaluated openings.",
    "Explain the concept of novelty in chess openings. Why do top players search for new moves?",
    "What is the best way to build an opening repertoire from scratch? Outline a step-by-step approach.",
    "Explain why understanding middlegame plans is more important than memorizing opening moves.",
    "What should a 1500-1800 ELO player focus on in their opening preparation?",
    "How do you learn from your opening mistakes? Describe a systematic approach to post-game analysis.",
    "What is the role of blitz chess in improving your opening knowledge? Is it helpful or harmful?",
    "Explain how to use a chess engine to analyze your opening play. What should you look for?",
    "What is the concept of a repertoire for White vs Black? Should they be consistent in style?",
    "Explain why playing 1.e4 is often recommended for beginners. What does it teach about chess?",
    "What is the concept of an opening tree? How do you organize your knowledge of variations?",
    "Explain how to handle an unfamiliar opening your opponent plays. What principles guide you?",
    "What is the best approach to learning a new opening? Study master games, read books, or use databases?",
    "Explain the concept of a universal system (like the London or KIA). Pros and cons for improvement?",
    "What is the relationship between opening knowledge and rating? At what level does it become critical?",
    "Explain how to identify your weaknesses in the opening phase. What patterns should you look for?",
    "What is the concept of a theoretical novelty (TN)? How deep do amateurs need to know theory?",
    "Explain the difference between learning an opening for correspondence chess vs rapid/blitz.",
    "What is the best way to prepare for a tournament? How much time should be spent on openings?",
    "Explain how to use opening databases (ChessBase, Lichess) to prepare your repertoire.",
    "What is the concept of a surprise weapon? How do you prepare a sideline for a specific opponent?",
    "Explain the psychological aspects of opening choice. How does your opponent's style affect your preparation?",
    "What is the concept of a drawing weapon? Which openings are good for playing for a draw with Black?",
    "Explain how to transition from the opening to the middlegame smoothly. What are the key checkpoints?",
    "What is the concept of piece coordination in the opening? Give 3 examples of good vs bad coordination.",
    "Explain the concept of prophylaxis in openings. How do you anticipate your opponent's plans?",
    "What opening advice would you give to a player returning to chess after many years?",
    "Explain how to study grandmaster games to improve your opening understanding.",
    "What is the concept of a theoretical draw? Which openings are most likely to lead to theoretical draws?",
    "Explain the relationship between time control and opening choice. How should your openings differ in rapid vs classical?",
    "What is the role of pattern recognition in opening play? How do you develop it?",
    "Explain the concept of a critical position in an opening. How do you identify the key moments?",
    "What opening resources (books, videos, databases) would you recommend for a 1000-1500 player?",
    "Explain how to practice openings: playing them in games, studying theory, or solving puzzles?",
    "What is the concept of opening fitness? How do you keep your repertoire up to date?",

    # =====================================================================
    # 9. HISTORICAL CONTEXT & FAMOUS GAMES (50)
    # =====================================================================
    "How did Bobby Fischer revolutionize the Sicilian Najdorf? What were his key contributions?",
    "Explain Kasparov's impact on the King's Indian Defense. What new ideas did he bring?",
    "What role did Anatoly Karpov play in developing the Caro-Kann at the highest level?",
    "How did the Kramnik-Kasparov 2000 match change the theory of the Berlin Defense?",
    "Explain Petrosian's contribution to the French Defense. What positional ideas did he demonstrate?",
    "What was Botvinnik's legacy in the Semi-Slav Defense and the English Opening?",
    "How did Morphy demonstrate the importance of rapid development in open games?",
    "Explain Steinitz's contribution to chess theory. How did he change how we think about openings?",
    "What was Nimzowitsch's hypermodern revolution? How did it change opening theory forever?",
    "How did Réti's 1.Nf3 challenge the classical school of Tarrasch?",
    "Explain the role of the Sicilian Defense in Fischer vs Spassky 1972. What variations were played?",
    "What opening innovations came from the Karpov-Kasparov matches? Name 3 key ideas.",
    "How did Anand contribute to the development of the Ruy Lopez at the highest level?",
    "Explain Carlsen's approach to openings. Why does he prefer solid, flexible setups?",
    "What was Marshall's contribution to the Ruy Lopez (the Marshall Attack)? Tell the historical story.",
    "How did the Polgar sisters influence opening theory in the 1990s?",
    "Explain Tal's sacrificial approach in the Sicilian. How did he create new attacking ideas?",
    "What was Alekhine's contribution to the defense that bears his name?",
    "How did Spassky handle the King's Gambit at the top level? Was it effective?",
    "Explain Geller's contributions to the King's Indian Defense and the Sicilian Najdorf.",
    "What opening innovations did Ding Liren bring to modern chess?",
    "How did computer engines change our understanding of the King's Indian Defense?",
    "Explain the role of the Petroff Defense in world championship matches throughout history.",
    "What was the significance of the game Morphy vs Duke of Brunswick (1858) for opening theory?",
    "How did the game Fischer vs Spassky, Game 6, 1972 (Queen's Gambit Declined) become legendary?",
    "Explain the significance of Kasparov vs Topalov, Wijk aan Zee 1999, for the Pirc Defense.",
    "What was the role of the Grünfeld Defense in world championship matches? Name key games.",
    "How did the Internet and online chess change opening preparation since 2000?",
    "Explain how AlphaZero's games in 2017 influenced opening theory. What new ideas were proposed?",
    "What was the significance of Anderssen's Immortal Game (1851) for gambit theory?",
    "How did Capablanca's simple style influence the popularity of the Queen's Gambit Declined?",
    "Explain Bronstein's creative contributions to the King's Indian Defense.",
    "What opening trends emerged from the Candidates Tournaments in recent years?",
    "How did the Najdorf become the most analyzed opening in history? Trace its development.",
    "Explain how Korchnoi used the French Defense as his main weapon for decades.",
    "What was Larsen's contribution to unorthodox openings (1.b3, the Scandinavian)?",
    "How did the development of tablebases affect opening preparation in endgame-related lines?",
    "Explain the role of secondants in opening preparation for world championship matches.",
    "What openings dominated the 2023-2024 Ding Liren vs Gukesh match? Analyze the choices.",
    "How did Firouzja's aggressive style influence modern opening trends among young players?",
    "Explain how the Catalan Opening became dominant in top-level chess from 2010 onwards.",
    "What was the significance of Deep Blue vs Kasparov (1997) for opening theory and preparation?",
    "How did the Berlin Defense transform from a rarity to a main weapon after 2000?",
    "Explain how Nakamura popularized certain sharp lines in online chess. How did this affect theory?",
    "What opening trends characterized the Candidates 2022 tournament? What was most played?",
    "How did Caruana prepare for the 2018 World Championship match? What opening strategy did he use?",
    "Explain how the London System went from a club player's weapon to a top GM choice.",
    "What impact did Leela Chess Zero have on opening theory? Compare its preferences with Stockfish.",
    "How has the Sicilian Sveshnikov evolved from Sveshnikov's original games to modern practice?",
    "Explain the historical debate about whether the King's Gambit is sound. What is the current verdict?",
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
