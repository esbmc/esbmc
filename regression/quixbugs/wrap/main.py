
def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = text.rfind(' ', 0, cols + 1)
        if end == -1:
            end = cols
        line, text = text[:end], text[end:]
        lines.append(line)

    lines.append(text)
    return lines


TEXT = ("The leaves did not stir on the trees, grasshoppers chirruped, and the "
        "monotonous hollow sound of the sea rising up from below, spoke of the "
        "peace, of the eternal sleep awaiting us. So it must have sounded when "
        "there was no Yalta, no Oreanda here; so it sounds now, and it will sound "
        "as indifferently and monotonously when we are all no more. And in this "
        "constancy, in this complete indifference to the life and death of each of "
        "us, there lies hid, perhaps, a pledge of our eternal salvation, of the "
        "unceasing movement of life upon earth, of unceasing progress towards "
        "perfection. Sitting beside a young woman who in the dawn seemed so lovely, "
        "soothed and spellbound in these magical surroundings - the sea, mountains, "
        "clouds, the open sky - Gurov thought how in reality everything is "
        "beautiful in this world when one reflects: everything except what we think "
        "or do ourselves when we forget our human dignity and the higher aims of "
        "our existence.")


assert wrap(TEXT, 50) == [
    "The leaves did not stir on the trees, grasshoppers",
    " chirruped, and the monotonous hollow sound of the",
    " sea rising up from below, spoke of the peace, of",
    " the eternal sleep awaiting us. So it must have",
    " sounded when there was no Yalta, no Oreanda here;",
    " so it sounds now, and it will sound as",
    " indifferently and monotonously when we are all no",
    " more. And in this constancy, in this complete",
    " indifference to the life and death of each of us,",
    " there lies hid, perhaps, a pledge of our eternal",
    " salvation, of the unceasing movement of life upon",
    " earth, of unceasing progress towards perfection.",
    " Sitting beside a young woman who in the dawn",
    " seemed so lovely, soothed and spellbound in these",
    " magical surroundings - the sea, mountains,",
    " clouds, the open sky - Gurov thought how in",
    " reality everything is beautiful in this world",
    " when one reflects: everything except what we",
    " think or do ourselves when we forget our human",
    " dignity and the higher aims of our existence."
]

assert wrap(TEXT, 20) == [
    "The leaves did not",
    " stir on the trees,",
    " grasshoppers",
    " chirruped, and the",
    " monotonous hollow",
    " sound of the sea",
    " rising up from",
    " below, spoke of the",
    " peace, of the",
    " eternal sleep",
    " awaiting us. So it",
    " must have sounded",
    " when there was no",
    " Yalta, no Oreanda",
    " here; so it sounds",
    " now, and it will",
    " sound as",
    " indifferently and",
    " monotonously when",
    " we are all no more.",
    " And in this",
    " constancy, in this",
    " complete",
    " indifference to the",
    " life and death of",
    " each of us, there",
    " lies hid, perhaps,",
    " a pledge of our",
    " eternal salvation,",
    " of the unceasing",
    " movement of life",
    " upon earth, of",
    " unceasing progress",
    " towards perfection.",
    " Sitting beside a",
    " young woman who in",
    " the dawn seemed so",
    " lovely, soothed and",
    " spellbound in these",
    " magical",
    " surroundings - the",
    " sea, mountains,",
    " clouds, the open",
    " sky - Gurov thought",
    " how in reality",
    " everything is",
    " beautiful in this",
    " world when one",
    " reflects:",
    " everything except",
    " what we think or do",
    " ourselves when we",
    " forget our human",
    " dignity and the",
    " higher aims of our",
    " existence."
]

assert wrap(TEXT, 80) == [
    "The leaves did not stir on the trees, grasshoppers chirruped, and the monotonous",
    " hollow sound of the sea rising up from below, spoke of the peace, of the",
    " eternal sleep awaiting us. So it must have sounded when there was no Yalta, no",
    " Oreanda here; so it sounds now, and it will sound as indifferently and",
    " monotonously when we are all no more. And in this constancy, in this complete",
    " indifference to the life and death of each of us, there lies hid, perhaps, a",
    " pledge of our eternal salvation, of the unceasing movement of life upon earth,",
    " of unceasing progress towards perfection. Sitting beside a young woman who in",
    " the dawn seemed so lovely, soothed and spellbound in these magical surroundings",
    " - the sea, mountains, clouds, the open sky - Gurov thought how in reality",
    " everything is beautiful in this world when one reflects: everything except what",
    " we think or do ourselves when we forget our human dignity and the higher aims",
    " of our existence."
]

assert wrap(TEXT, 77) == [
    "The leaves did not stir on the trees, grasshoppers chirruped, and the",
    " monotonous hollow sound of the sea rising up from below, spoke of the peace,",
    " of the eternal sleep awaiting us. So it must have sounded when there was no",
    " Yalta, no Oreanda here; so it sounds now, and it will sound as indifferently",
    " and monotonously when we are all no more. And in this constancy, in this",
    " complete indifference to the life and death of each of us, there lies hid,",
    " perhaps, a pledge of our eternal salvation, of the unceasing movement of",
    " life upon earth, of unceasing progress towards perfection. Sitting beside a",
    " young woman who in the dawn seemed so lovely, soothed and spellbound in",
    " these magical surroundings - the sea, mountains, clouds, the open sky -",
    " Gurov thought how in reality everything is beautiful in this world when one",
    " reflects: everything except what we think or do ourselves when we forget our",
    " human dignity and the higher aims of our existence."
]

assert wrap(TEXT, 140) == [
    "The leaves did not stir on the trees, grasshoppers chirruped, and the monotonous hollow sound of the sea rising up from below, spoke of the",
    " peace, of the eternal sleep awaiting us. So it must have sounded when there was no Yalta, no Oreanda here; so it sounds now, and it will",
    " sound as indifferently and monotonously when we are all no more. And in this constancy, in this complete indifference to the life and death",
    " of each of us, there lies hid, perhaps, a pledge of our eternal salvation, of the unceasing movement of life upon earth, of unceasing",
    " progress towards perfection. Sitting beside a young woman who in the dawn seemed so lovely, soothed and spellbound in these magical",
    " surroundings - the sea, mountains, clouds, the open sky - Gurov thought how in reality everything is beautiful in this world when one",
    " reflects: everything except what we think or do ourselves when we forget our human dignity and the higher aims of our existence."
]

