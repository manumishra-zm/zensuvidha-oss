"""Grounding guard — the layer that checks the model instead of trusting it.

Prompt rules alone do not hold a 3–4B local model. Asked for a service the business
doesn't offer it will invent a fee; asked the distance to Mumbai it will answer in
kilometres. The failure is far worse in Hindi/Telugu/Tamil, where the model is weaker,
drifts between languages, and runs out of token budget mid-word — which a caller hears
as garbled nonsense.

So the engine verifies every reply before a single word is spoken:

  * **Grounding** — every number in a reply (₹ fees, phone numbers, timings, distances)
    must appear in the pack's own facts or in something the caller themselves said.
    A fabricated "₹1,000" is caught and never reaches the speaker.
  * **Language** — the reply must actually be written in the language the caller is
    speaking, not a different Indic script or a fall-back into English.
  * **Truncation** — a reply cut off mid-word by the token budget is trimmed back to
    its last complete sentence instead of being spoken as a fragment.

When a check fails, the reply is replaced with a pre-written line in the caller's own
language — "I don't have that detail, let me confirm with the desk", or "I'm the
receptionist here, please ask me about our services". The model decides *that* it
can't answer; this module supplies the *words*, because a small model writing a
refusal in Telugu is exactly where it garbles.
"""
import re

# --------------------------------------------------------------------------- #
# Digits — Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu,
# Kannada, Malayalam, Arabic-Indic and Extended Arabic-Indic → ASCII, so
# "₹1,500", "१५००" and "౧౫౦౦" all compare as the same fact.
# --------------------------------------------------------------------------- #
_DIGIT_BASES = (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6,
                0x0C66, 0x0CE6, 0x0D66, 0x0660, 0x06F0)
_DIGIT_MAP = {base + i: str(i) for base in _DIGIT_BASES for i in range(10)}

# a run of digits that may contain grouping separators: "1,500", "1234 5678", "20–40"
_SPAN_RE = re.compile(r"\d[\d,٫\s\-–—]*\d|\d")
_RUN_RE = re.compile(r"\d+(?:\.\d+)?")


def norm_digits(text: str) -> str:
    """Indic digits → ASCII digits."""
    return (text or "").translate(_DIGIT_MAP)


def _canon(n: str) -> str:
    """1500.00 → 1500, 007 → 7, so formatting differences aren't false alarms."""
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n.lstrip("0") or "0"


def numbers_in(text: str) -> set:
    """Every number in `text`, normalised.

    Each digit span contributes both its individual runs and the runs joined
    together, so a phone written "+91 20 1234 5678" in the pack still matches
    "912012345678" spoken back by the model (and vice versa).
    """
    t = norm_digits(text)
    out = set()
    for span in _SPAN_RE.findall(t):
        # a comma inside a number is a thousands separator, not a boundary: ₹1,500 and
        # ₹1,50,000 are single values, so join across it before splitting.
        runs = _RUN_RE.findall(re.sub(r"(?<=\d)[,٫](?=\d)", "", span))
        for r in runs:
            out.add(_canon(r))
        if len(runs) > 1:            # spaced/dashed groups → also the run-together form
            out.add(_canon("".join(runs)))
    return out


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield str(obj)


_PACK_NUM_CACHE: dict = {}


def pack_numbers(pack: dict) -> frozenset:
    """Every number that legitimately appears anywhere in an industry pack —
    prices, hours, phone numbers, years, package rates, waiting times."""
    key = pack.get("id")
    if key and key in _PACK_NUM_CACHE:
        return _PACK_NUM_CACHE[key]
    nums = set()
    for s in _walk_strings(pack):
        nums |= numbers_in(s)
    frozen = frozenset(nums)
    if key:
        _PACK_NUM_CACHE[key] = frozen
    return frozen


def ungrounded_numbers(reply: str, allowed) -> list:
    """Numbers the model stated that are backed by nothing — the fabricated facts."""
    return sorted(numbers_in(reply) - set(allowed))


# --------------------------------------------------------------------------- #
# Scripts — used to check the reply is in the language the caller is speaking
# --------------------------------------------------------------------------- #
_SCRIPT_RANGES = [
    (0x0900, 0x097F, "Devanagari"),   # Hindi, Marathi
    (0x0980, 0x09FF, "Bengali"),      # Bengali, Assamese
    (0x0A00, 0x0A7F, "Gurmukhi"),     # Punjabi
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0600, 0x06FF, "Arabic"),       # Urdu
]
# How many tokens the SAME sentence costs in each language, relative to English, on
# Qwen's tokenizer. Measured, not guessed — one sentence of clinic opening hours:
#   English 27 · Urdu 68 · Marathi 90 · Hindi 94 · Bengali 106 · Gujarati 119
#   Tamil 123 · Malayalam 132 · Kannada 138 · Telugu 168 · Odia 191
# Rounded UP, because under-budgeting is what cuts a reply off mid-word. This is a
# ceiling on the reply, not a target — the model still stops when it is done.
TOKEN_FACTOR = {
    "Urdu": 3, "Marathi": 4, "Hindi": 4, "Bengali": 4, "Assamese": 4,
    "Gujarati": 5, "Tamil": 5, "Punjabi": 5, "Malayalam": 5,
    "Kannada": 6, "Telugu": 7, "Odia": 8,
}

# language name (as used by the orchestrator) → the script it is written in
LANG_SCRIPT = {
    "Hindi": "Devanagari", "Marathi": "Devanagari",
    "Bengali": "Bengali", "Assamese": "Bengali",
    "Punjabi": "Gurmukhi", "Gujarati": "Gujarati", "Odia": "Odia",
    "Tamil": "Tamil", "Telugu": "Telugu", "Kannada": "Kannada",
    "Malayalam": "Malayalam", "Urdu": "Arabic",
}


def script_share(text: str, script: str) -> float:
    """Fraction of the reply's letters written in `script` (0.0–1.0).

    Latin letters count against it, so a Telugu reply that quietly falls back to
    English scores low — but a couple of Latin proper nouns ("Dr Rao", "UPI") in
    an otherwise Telugu sentence still scores high.
    """
    total = hits = 0
    for c in text or "":
        if not c.isalpha():
            continue
        total += 1
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= ord(c) <= hi:
                if name == script:
                    hits += 1
                break
    return (hits / total) if total else 1.0


# --------------------------------------------------------------------------- #
# Safe replies — pre-written so the caller always hears clean, natural speech
# in their own language, even when the model itself could not produce it.
#
#   unknown  → we simply don't hold that fact
#   scope    → the caller asked for something this receptionist doesn't do
#   repeat   → we couldn't make out what was said
#   ref      → the booking reference, so a Hindi call doesn't end in an English sentence
#
# Wording is feminine-first to match the default female pack voices (Tara, Lekha…).
# For a male persona, edit the verb forms here — nothing else depends on them.
# --------------------------------------------------------------------------- #
SAFE_LINES = {
    "English": {
        "still_there": "Are you still there? I'm happy to wait a moment.",
        "goodbye": "I'll let you go for now — please call back any time. Goodbye.",
        "unknown": "I don't have that detail with me — let me confirm it with the front desk. "
                   "Is there anything else I can help you with?",
        "scope": "I'm the receptionist at {biz} — I can help with appointments, timings, fees "
                 "and our services. Please ask me about that.",
        "repeat": "Sorry, I didn't quite catch that — could you say it once more?",
        "noisy": "There's a lot of background noise coming through — could you move somewhere quieter and say that again?",
        "faint": "The line isn't very clear — could you say that once more?",
        "ref": "Your booking reference is #{ref}.",
        "go_on": "Go on, I'm listening.",
    },
    "Hindi": {
        "still_there": "क्या आप लाइन पर हैं? मैं थोड़ी देर इंतज़ार कर सकती हूँ।",
        "goodbye": "अभी के लिए मैं कॉल बंद कर रही हूँ — कभी भी दोबारा कॉल कीजिए। नमस्ते।",
        "unknown": "माफ़ कीजिए, यह जानकारी मेरे पास नहीं है। मैं डेस्क से पुष्टि करके बता दूँगी। "
                   "क्या मैं आपकी और किसी बात में मदद कर सकती हूँ?",
        "scope": "मैं {biz} की रिसेप्शनिस्ट हूँ। मैं अपॉइंटमेंट, समय, फ़ीस और हमारी सेवाओं के बारे में "
                 "मदद कर सकती हूँ। कृपया इसी बारे में पूछिए।",
        "repeat": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाई। कृपया एक बार फिर कहिए।",
        "noisy": "पीछे काफ़ी शोर आ रहा है — क्या आप किसी शांत जगह से दोबारा कहेंगे?",
        "faint": "लाइन साफ़ नहीं आ रही — कृपया एक बार फिर कहिए।",
        "ref": "आपका बुकिंग रेफरेंस #{ref} है।",
        "go_on": "जी कहिए, मैं सुन रही हूँ।",
    },
    "Telugu": {
        "still_there": "మీరు లైన్‌లో ఉన్నారా? నేను కొంచెం సేపు వేచి ఉంటాను.",
        "goodbye": "ప్రస్తుతానికి కాల్ ముగిస్తున్నాను — ఎప్పుడైనా మళ్ళీ కాల్ చేయండి. ధన్యవాదాలు.",
        "unknown": "క్షమించండి, ఆ వివరం నా దగ్గర లేదు. నేను డెస్క్‌లో కనుక్కుని చెబుతాను. "
                   "ఇంకేమైనా సహాయం కావాలా?",
        "scope": "నేను {biz} రిసెప్షనిస్ట్‌ని. అపాయింట్‌మెంట్లు, సమయాలు, ఫీజులు మరియు మా సేవల గురించి "
                 "మాత్రమే సహాయం చేయగలను. దయచేసి వాటి గురించి అడగండి.",
        "repeat": "క్షమించండి, సరిగ్గా వినిపించలేదు. దయచేసి మరోసారి చెప్పండి.",
        "noisy": "వెనుక చాలా శబ్దం వస్తోంది — కొంచెం నిశ్శబ్దమైన చోటి నుంచి మళ్ళీ చెప్పగలరా?",
        "faint": "లైన్ స్పష్టంగా లేదు — దయచేసి మరోసారి చెప్పండి.",
        "ref": "మీ బుకింగ్ రిఫరెన్స్ #{ref}.",
        "go_on": "చెప్పండి, నేను వింటున్నాను.",
    },
    "Tamil": {
        "still_there": "நீங்கள் இணைப்பில் இருக்கிறீர்களா? சிறிது நேரம் காத்திருக்கிறேன்.",
        "goodbye": "இப்போதைக்கு அழைப்பை முடிக்கிறேன் — எப்போது வேண்டுமானாலும் அழைக்கவும். நன்றி.",
        "unknown": "மன்னிக்கவும், அந்த விவரம் என்னிடம் இல்லை. நான் முன்பதிவு மேசையில் கேட்டு உறுதிப்படுத்துகிறேன். "
                   "வேறு ஏதேனும் உதவி வேண்டுமா?",
        "scope": "நான் {biz} வரவேற்பாளர். சந்திப்பு, நேரம், கட்டணம் மற்றும் எங்கள் சேவைகள் பற்றி மட்டுமே "
                 "உதவ முடியும். தயவுசெய்து அதைப் பற்றி கேளுங்கள்.",
        "repeat": "மன்னிக்கவும், சரியாகக் கேட்கவில்லை. தயவுசெய்து மீண்டும் ஒருமுறை சொல்லுங்கள்.",
        "noisy": "பின்னணியில் அதிக சத்தம் கேட்கிறது — அமைதியான இடத்திலிருந்து மீண்டும் சொல்ல முடியுமா?",
        "faint": "இணைப்பு தெளிவாக இல்லை — தயவுசெய்து மீண்டும் ஒருமுறை சொல்லுங்கள்.",
        "ref": "உங்கள் முன்பதிவு குறிப்பு எண் #{ref}.",
        "go_on": "சொல்லுங்கள், நான் கேட்கிறேன்.",
    },
    "Kannada": {
        "still_there": "ನೀವು ಲೈನ್‌ನಲ್ಲಿ ಇದ್ದೀರಾ? ಸ್ವಲ್ಪ ಹೊತ್ತು ಕಾಯುತ್ತೇನೆ.",
        "goodbye": "ಸದ್ಯಕ್ಕೆ ಕರೆ ಮುಗಿಸುತ್ತೇನೆ — ಯಾವಾಗ ಬೇಕಾದರೂ ಮತ್ತೆ ಕರೆ ಮಾಡಿ. ಧನ್ಯವಾದ.",
        "unknown": "ಕ್ಷಮಿಸಿ, ಆ ವಿವರ ನನ್ನ ಬಳಿ ಇಲ್ಲ. ನಾನು ಡೆಸ್ಕ್‌ನಲ್ಲಿ ಕೇಳಿ ತಿಳಿಸುತ್ತೇನೆ. "
                   "ಬೇರೆ ಏನಾದರೂ ಸಹಾಯ ಬೇಕೆ?",
        "scope": "ನಾನು {biz} ಸ್ವಾಗತಕಾರಿ. ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್, ಸಮಯ, ಶುಲ್ಕ ಮತ್ತು ನಮ್ಮ ಸೇವೆಗಳ ಬಗ್ಗೆ ಮಾತ್ರ "
                 "ಸಹಾಯ ಮಾಡಬಲ್ಲೆ. ದಯವಿಟ್ಟು ಅದರ ಬಗ್ಗೆ ಕೇಳಿ.",
        "repeat": "ಕ್ಷಮಿಸಿ, ಸರಿಯಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಇನ್ನೊಮ್ಮೆ ಹೇಳಿ.",
        "noisy": "ಹಿನ್ನೆಲೆಯಲ್ಲಿ ತುಂಬಾ ಶಬ್ದವಿದೆ — ಸ್ವಲ್ಪ ಶಾಂತವಾದ ಸ್ಥಳದಿಂದ ಮತ್ತೊಮ್ಮೆ ಹೇಳಬಹುದೇ?",
        "faint": "ಲೈನ್ ಸ್ಪಷ್ಟವಾಗಿಲ್ಲ — ದಯವಿಟ್ಟು ಇನ್ನೊಮ್ಮೆ ಹೇಳಿ.",
        "ref": "ನಿಮ್ಮ ಬುಕಿಂಗ್ ಉಲ್ಲೇಖ ಸಂಖ್ಯೆ #{ref}.",
        "go_on": "ಹೇಳಿ, ನಾನು ಕೇಳುತ್ತಿದ್ದೇನೆ.",
    },
    "Malayalam": {
        "still_there": "നിങ്ങൾ ലൈനിൽ ഉണ്ടോ? ഞാൻ കുറച്ചു നേരം കാത്തിരിക്കാം.",
        "goodbye": "തൽക്കാലം കോൾ അവസാനിപ്പിക്കുന്നു — എപ്പോൾ വേണമെങ്കിലും വിളിക്കൂ. നന്ദി.",
        "unknown": "ക്ഷമിക്കണം, ആ വിവരം എന്റെ പക്കലില്ല. ഞാൻ ഡെസ്കിൽ ചോദിച്ച് അറിയിക്കാം. "
                   "മറ്റെന്തെങ്കിലും സഹായം വേണോ?",
        "scope": "ഞാൻ {biz} ലെ റിസപ്ഷനിസ്റ്റാണ്. അപ്പോയിന്റ്മെന്റ്, സമയം, ഫീസ്, ഞങ്ങളുടെ സേവനങ്ങൾ "
                 "എന്നിവയിൽ മാത്രമേ സഹായിക്കാൻ കഴിയൂ. ദയവായി അതിനെക്കുറിച്ച് ചോദിക്കൂ.",
        "repeat": "ക്ഷമിക്കണം, ശരിക്ക് കേട്ടില്ല. ദയവായി ഒന്നുകൂടി പറയാമോ?",
        "noisy": "പിന്നിൽ വളരെയധികം ശബ്ദമുണ്ട് — ശാന്തമായ ഒരിടത്തുനിന്ന് ഒന്നുകൂടി പറയാമോ?",
        "faint": "ലൈൻ വ്യക്തമല്ല — ദയവായി ഒന്നുകൂടി പറയൂ.",
        "ref": "നിങ്ങളുടെ ബുക്കിംഗ് റഫറൻസ് #{ref}.",
        "go_on": "പറയൂ, ഞാൻ കേൾക്കുന്നു.",
    },
    "Bengali": {
        "still_there": "আপনি কি লাইনে আছেন? আমি একটু অপেক্ষা করছি।",
        "goodbye": "এখনকার মতো কল শেষ করছি — যে কোনো সময় আবার ফোন করবেন। ধন্যবাদ।",
        "unknown": "দুঃখিত, এই তথ্যটি আমার কাছে নেই। আমি ডেস্কে জেনে জানাচ্ছি। "
                   "আর কিছু সাহায্য লাগবে?",
        "scope": "আমি {biz}-এর রিসেপশনিস্ট। আমি অ্যাপয়েন্টমেন্ট, সময়, ফি এবং আমাদের পরিষেবা নিয়ে "
                 "সাহায্য করতে পারি। অনুগ্রহ করে সেই বিষয়েই জিজ্ঞাসা করুন।",
        "repeat": "দুঃখিত, ঠিকমতো শুনতে পাইনি। অনুগ্রহ করে আরেকবার বলুন।",
        "noisy": "পিছনে অনেক আওয়াজ হচ্ছে — একটু শান্ত জায়গা থেকে আবার বলবেন?",
        "faint": "লাইন পরিষ্কার আসছে না — অনুগ্রহ করে আরেকবার বলুন।",
        "ref": "আপনার বুকিং রেফারেন্স #{ref}।",
        "go_on": "বলুন, আমি শুনছি।",
    },
    "Marathi": {
        "still_there": "तुम्ही लाइनवर आहात का? मी थोडा वेळ थांबते.",
        "goodbye": "आत्तापुरता कॉल बंद करते — कधीही पुन्हा फोन करा. धन्यवाद.",
        "unknown": "माफ करा, ती माहिती माझ्याकडे नाही. मी डेस्कवर विचारून सांगते. "
                   "आणखी काही मदत हवी आहे का?",
        "scope": "मी {biz} ची रिसेप्शनिस्ट आहे. मी अपॉइंटमेंट, वेळ, फी आणि आमच्या सेवांबद्दल मदत करू शकते. "
                 "कृपया त्याबद्दलच विचारा.",
        "repeat": "माफ करा, नीट ऐकू आलं नाही. कृपया पुन्हा एकदा सांगा.",
        "noisy": "मागे खूप आवाज येतोय — जरा शांत ठिकाणाहून पुन्हा सांगाल का?",
        "faint": "लाइन स्पष्ट येत नाही — कृपया पुन्हा एकदा सांगा.",
        "ref": "तुमचा बुकिंग संदर्भ क्रमांक #{ref} आहे.",
        "go_on": "सांगा, मी ऐकते आहे.",
    },
    "Gujarati": {
        "still_there": "શું તમે લાઇન પર છો? હું થોડી વાર રાહ જોઉં છું.",
        "goodbye": "હમણાં માટે કૉલ પૂરો કરું છું — ગમે ત્યારે ફરી ફોન કરજો. આભાર.",
        "unknown": "માફ કરશો, એ માહિતી મારી પાસે નથી. હું ડેસ્ક પર પૂછીને જણાવીશ. "
                   "બીજી કોઈ મદદ જોઈએ છે?",
        "scope": "હું {biz} ની રિસેપ્શનિસ્ટ છું. હું એપોઇન્ટમેન્ટ, સમય, ફી અને અમારી સેવાઓ વિશે મદદ કરી શકું છું. "
                 "કૃપા કરીને એ વિશે પૂછો.",
        "repeat": "માફ કરશો, બરાબર સંભળાયું નહીં. કૃપા કરીને ફરી એકવાર કહો.",
        "noisy": "પાછળ ઘણો અવાજ આવે છે — થોડી શાંત જગ્યાએથી ફરી કહેશો?",
        "faint": "લાઇન સ્પષ્ટ નથી — કૃપા કરીને ફરી એકવાર કહો.",
        "ref": "તમારો બુકિંગ સંદર્ભ #{ref} છે.",
        "go_on": "કહો, હું સાંભળું છું.",
    },
    "Punjabi": {
        "still_there": "ਕੀ ਤੁਸੀਂ ਲਾਈਨ ਉੱਤੇ ਹੋ? ਮੈਂ ਥੋੜ੍ਹੀ ਦੇਰ ਉਡੀਕਦੀ ਹਾਂ।",
        "goodbye": "ਹੁਣ ਲਈ ਕਾਲ ਬੰਦ ਕਰ ਰਹੀ ਹਾਂ — ਕਦੇ ਵੀ ਦੁਬਾਰਾ ਫ਼ੋਨ ਕਰੋ। ਧੰਨਵਾਦ।",
        "unknown": "ਮਾਫ਼ ਕਰਨਾ, ਇਹ ਜਾਣਕਾਰੀ ਮੇਰੇ ਕੋਲ ਨਹੀਂ ਹੈ। ਮੈਂ ਡੈਸਕ ਤੋਂ ਪੁੱਛ ਕੇ ਦੱਸਾਂਗੀ। "
                   "ਹੋਰ ਕੋਈ ਮਦਦ ਚਾਹੀਦੀ ਹੈ?",
        "scope": "ਮੈਂ {biz} ਦੀ ਰਿਸੈਪਸ਼ਨਿਸਟ ਹਾਂ। ਮੈਂ ਅਪਾਇੰਟਮੈਂਟ, ਸਮਾਂ, ਫ਼ੀਸ ਅਤੇ ਸਾਡੀਆਂ ਸੇਵਾਵਾਂ ਬਾਰੇ ਮਦਦ ਕਰ ਸਕਦੀ ਹਾਂ। "
                 "ਕਿਰਪਾ ਕਰਕੇ ਉਸੇ ਬਾਰੇ ਪੁੱਛੋ।",
        "repeat": "ਮਾਫ਼ ਕਰਨਾ, ਠੀਕ ਤਰ੍ਹਾਂ ਸੁਣਾਈ ਨਹੀਂ ਦਿੱਤਾ। ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਵਾਰ ਫਿਰ ਕਹੋ।",
        "noisy": "ਪਿੱਛੇ ਬਹੁਤ ਰੌਲਾ ਆ ਰਿਹਾ ਹੈ — ਕੀ ਤੁਸੀਂ ਕਿਸੇ ਸ਼ਾਂਤ ਥਾਂ ਤੋਂ ਦੁਬਾਰਾ ਕਹੋਗੇ?",
        "faint": "ਲਾਈਨ ਸਾਫ਼ ਨਹੀਂ ਆ ਰਹੀ — ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਵਾਰ ਫਿਰ ਕਹੋ।",
        "ref": "ਤੁਹਾਡਾ ਬੁਕਿੰਗ ਰੈਫਰੈਂਸ #{ref} ਹੈ।",
        "go_on": "ਦੱਸੋ, ਮੈਂ ਸੁਣ ਰਹੀ ਹਾਂ।",
    },
    "Urdu": {
        "still_there": "کیا آپ لائن پر ہیں؟ میں تھوڑی دیر انتظار کر لیتی ہوں۔",
        "goodbye": "ابھی کے لیے کال بند کر رہی ہوں — کسی بھی وقت دوبارہ فون کیجیے۔ شکریہ۔",
        "unknown": "معاف کیجیے، یہ معلومات میرے پاس نہیں ہیں۔ میں ڈیسک سے پوچھ کر بتاتی ہوں۔ "
                   "کیا کوئی اور مدد چاہیے؟",
        "scope": "میں {biz} کی ریسپشنسٹ ہوں۔ میں اپائنٹمنٹ، اوقات، فیس اور ہماری خدمات کے بارے میں "
                 "مدد کر سکتی ہوں۔ براہ کرم اسی بارے میں پوچھیے۔",
        "repeat": "معاف کیجیے، ٹھیک سے سنائی نہیں دیا۔ براہ کرم ایک بار پھر کہیے۔",
        "noisy": "پیچھے کافی شور آ رہا ہے — کیا آپ کسی پرسکون جگہ سے دوبارہ کہیں گے؟",
        "faint": "لائن صاف نہیں آ رہی — براہِ کرم ایک بار پھر کہیے۔",
        "ref": "آپ کا بکنگ ریفرنس #{ref} ہے۔",
        "go_on": "جی کہیے، میں سن رہی ہوں۔",
    },
    "Odia": {
        "still_there": "ଆପଣ ଲାଇନରେ ଅଛନ୍ତି କି? ମୁଁ ଟିକେ ଅପେକ୍ଷା କରୁଛି।",
        "goodbye": "ଏବେ ପାଇଁ କଲ୍ ଶେଷ କରୁଛି — ଯେକୌଣସି ସମୟରେ ପୁଣି ଫୋନ୍ କରନ୍ତୁ। ଧନ୍ୟବାଦ।",
        "unknown": "କ୍ଷମା କରନ୍ତୁ, ସେ ସୂଚନା ମୋ ପାଖରେ ନାହିଁ। ମୁଁ ଡେସ୍କରୁ ପଚାରି ଜଣାଇବି। "
                   "ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର କି?",
        "scope": "ମୁଁ {biz} ର ରିସେପ୍ସନିଷ୍ଟ। ମୁଁ ଆପଏଣ୍ଟମେଣ୍ଟ, ସମୟ, ଫି ଏବଂ ଆମର ସେବା ବିଷୟରେ ସାହାଯ୍ୟ କରିପାରିବି। "
                 "ଦୟାକରି ସେ ବିଷୟରେ ପଚାରନ୍ତୁ।",
        "repeat": "କ୍ଷମା କରନ୍ତୁ, ଠିକ୍ ଭାବରେ ଶୁଣିପାରିଲି ନାହିଁ। ଦୟାକରି ଆଉ ଥରେ କୁହନ୍ତୁ।",
        "noisy": "ପଛରେ ବହୁତ ଶବ୍ଦ ଆସୁଛି — ଟିକେ ଶାନ୍ତ ଜାଗାରୁ ପୁଣି କହିପାରିବେ କି?",
        "faint": "ଲାଇନ ସ୍ପଷ୍ଟ ଆସୁନାହିଁ — ଦୟାକରି ଆଉ ଥରେ କୁହନ୍ତୁ।",
        "ref": "ଆପଣଙ୍କ ବୁକିଂ ରେଫରେନ୍ସ #{ref}।",
        "go_on": "କୁହନ୍ତୁ, ମୁଁ ଶୁଣୁଛି।",
    },
}

# Romanised Indian languages (Hinglish and friends) — the caller is typing/speaking
# an Indian language in Latin letters, so the reply must stay in Latin letters too.
SAFE_LINES_ROMAN = {
    "Hindi": {
        "unknown": "Sorry, yeh jaankari mere paas nahi hai. Main desk se confirm karke bata dungi. "
                   "Aur kuch help chahiye?",
        "scope": "Main {biz} ki receptionist hoon. Main appointment, timing, fees aur hamari services "
                 "ke baare mein help kar sakti hoon. Please usi baare mein poochiye.",
        "repeat": "Sorry, main theek se sun nahi payi. Please ek baar phir kahiye.",
        "noisy": "Peechhe kaafi shor aa raha hai — kya aap kisi shaant jagah se dobara kahenge?",
        "faint": "Line saaf nahi aa rahi — kripya ek baar phir kahiye.",
        "ref": "Aapka booking reference #{ref} hai.",
        "ask_name": "Aapka pura naam bata dijiye.",
        "ask_phone": "Aapka mobile number kya hai?",
        "ask_datetime": "Aapko kaun sa din aur kaun sa time chahiye?",
        "ask_service": "Aap kis cheez ke liye appointment lena chahenge?",
        "ask_confirm": "Aapki saari details mil gayi hain. Kya main appointment book kar doon?",
        "go_on": "Ji kahiye, main sun rahi hoon.",
    },
}

# --------------------------------------------------------------------------- #
# Booking questions, pre-written per language
# --------------------------------------------------------------------------- #
# Same reasoning as SAFE_LINES: a 4B model asked to compose "which day and time suits
# you?" in Telugu mid-call produces something worse than a sentence we wrote once. These
# are the recovery line when a reply is rejected DURING an active booking — asking for
# what's still missing keeps the call moving, where "I'll check with the desk" derails it.
# A pack may override any of these per language (booking.slots.<field>_hi, …).
ASK_LINES = {
    "English":   {"name": "May I have your full name, please?",
                  "phone": "What mobile number should we use?",
                  "datetime": "Which day and time would suit you?",
                  "service": "What would you like to book?",
                  "confirm": "I have all your details. Shall I confirm the booking?"},
    "Hindi":     {"name": "आपका पूरा नाम बता दीजिए।",
                  "phone": "आपका मोबाइल नंबर क्या है?",
                  "datetime": "आपको कौन सा दिन और कौन सा समय चाहिए?",
                  "service": "आप किस चीज़ के लिए अपॉइंटमेंट लेना चाहेंगे?",
                  "confirm": "मेरे पास आपकी सारी जानकारी है। क्या मैं अपॉइंटमेंट बुक कर दूँ?"},
    "Telugu":    {"name": "మీ పూర్తి పేరు చెప్పగలరా?",
                  "phone": "మీ మొబైల్ నంబర్ చెప్పండి.",
                  "datetime": "మీకు ఏ రోజు, ఏ సమయం అనుకూలంగా ఉంటుంది?",
                  "service": "మీరు దేని కోసం అపాయింట్‌మెంట్ కావాలనుకుంటున్నారు?",
                  "confirm": "మీ వివరాలన్నీ నా దగ్గర ఉన్నాయి. అపాయింట్‌మెంట్ బుక్ చేయమంటారా?"},
    "Tamil":     {"name": "உங்கள் முழுப் பெயரைச் சொல்ல முடியுமா?",
                  "phone": "உங்கள் மொபைல் எண்ணைச் சொல்லுங்கள்.",
                  "datetime": "உங்களுக்கு எந்த நாள், எந்த நேரம் வசதியாக இருக்கும்?",
                  "service": "நீங்கள் எதற்காக நேரம் பதிவு செய்ய விரும்புகிறீர்கள்?",
                  "confirm": "உங்கள் விவரங்கள் அனைத்தும் கிடைத்துவிட்டன. நேரத்தைப் பதிவு செய்யலாமா?"},
    "Kannada":   {"name": "ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರು ಹೇಳಬಹುದೇ?",
                  "phone": "ನಿಮ್ಮ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ ಹೇಳಿ.",
                  "datetime": "ನಿಮಗೆ ಯಾವ ದಿನ ಮತ್ತು ಯಾವ ಸಮಯ ಅನುಕೂಲ?",
                  "service": "ನೀವು ಯಾವುದಕ್ಕಾಗಿ ಸಮಯ ಕಾಯ್ದಿರಿಸಲು ಬಯಸುತ್ತೀರಿ?",
                  "confirm": "ನಿಮ್ಮ ಎಲ್ಲಾ ವಿವರಗಳು ಸಿಕ್ಕಿವೆ. ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬುಕ್ ಮಾಡಲೇ?"},
    "Malayalam": {"name": "താങ്കളുടെ മുഴുവൻ പേര് പറയാമോ?",
                  "phone": "താങ്കളുടെ മൊബൈൽ നമ്പർ പറയൂ.",
                  "datetime": "ഏത് ദിവസവും ഏത് സമയവുമാണ് സൗകര്യപ്രദം?",
                  "service": "എന്തിനുവേണ്ടിയാണ് സമയം ബുക്ക് ചെയ്യേണ്ടത്?",
                  "confirm": "താങ്കളുടെ എല്ലാ വിവരങ്ങളും കിട്ടി. അപ്പോയിന്റ്മെന്റ് ബുക്ക് ചെയ്യട്ടെ?"},
    "Bengali":   {"name": "আপনার পুরো নামটা বলবেন?",
                  "phone": "আপনার মোবাইল নম্বরটা বলুন।",
                  "datetime": "কোন দিন আর কোন সময় আপনার সুবিধা হবে?",
                  "service": "আপনি কীসের জন্য সময় নিতে চান?",
                  "confirm": "আপনার সব তথ্য পেয়েছি। অ্যাপয়েন্টমেন্ট বুক করে দেব?"},
    "Marathi":   {"name": "तुमचं पूर्ण नाव सांगाल का?",
                  "phone": "तुमचा मोबाईल नंबर सांगा.",
                  "datetime": "तुम्हाला कोणता दिवस आणि कोणती वेळ सोयीची आहे?",
                  "service": "तुम्हाला कशासाठी अपॉइंटमेंट हवी आहे?",
                  "confirm": "तुमची सर्व माहिती मिळाली. अपॉइंटमेंट बुक करू का?"},
    "Gujarati":  {"name": "તમારું પૂરું નામ કહેશો?",
                  "phone": "તમારો મોબાઇલ નંબર જણાવો.",
                  "datetime": "તમને કયો દિવસ અને કયો સમય અનુકૂળ રહેશે?",
                  "service": "તમારે શેના માટે અપોઇન્ટમેન્ટ જોઈએ છે?",
                  "confirm": "તમારી બધી માહિતી મળી ગઈ. અપોઇન્ટમેન્ટ બુક કરી દઉં?"},
    "Punjabi":   {"name": "ਤੁਹਾਡਾ ਪੂਰਾ ਨਾਂ ਦੱਸੋਗੇ?",
                  "phone": "ਤੁਹਾਡਾ ਮੋਬਾਈਲ ਨੰਬਰ ਦੱਸੋ।",
                  "datetime": "ਤੁਹਾਨੂੰ ਕਿਹੜਾ ਦਿਨ ਅਤੇ ਕਿਹੜਾ ਸਮਾਂ ਠੀਕ ਰਹੇਗਾ?",
                  "service": "ਤੁਸੀਂ ਕਿਸ ਲਈ ਸਮਾਂ ਲੈਣਾ ਚਾਹੁੰਦੇ ਹੋ?",
                  "confirm": "ਤੁਹਾਡੀ ਸਾਰੀ ਜਾਣਕਾਰੀ ਮਿਲ ਗਈ ਹੈ। ਅਪਾਇੰਟਮੈਂਟ ਬੁੱਕ ਕਰ ਦਿਆਂ?"},
    "Urdu":      {"name": "آپ کا پورا نام بتا دیجیے۔",
                  "phone": "آپ کا موبائل نمبر بتائیے۔",
                  "datetime": "آپ کو کون سا دن اور کون سا وقت مناسب رہے گا؟",
                  "service": "آپ کس چیز کے لیے وقت لینا چاہتے ہیں؟",
                  "confirm": "آپ کی تمام تفصیلات مل گئی ہیں۔ کیا میں اپائنٹمنٹ بک کر دوں؟"},
    "Odia":      {"name": "ଆପଣଙ୍କ ପୂରା ନାମ କହିବେ କି?",
                  "phone": "ଆପଣଙ୍କ ମୋବାଇଲ ନମ୍ବର କୁହନ୍ତୁ।",
                  "datetime": "ଆପଣଙ୍କୁ କେଉଁ ଦିନ ଏବଂ କେଉଁ ସମୟ ସୁବିଧା ହେବ?",
                  "service": "ଆପଣ କାହିଁ ପାଇଁ ସମୟ ନେବାକୁ ଚାହୁଁଛନ୍ତି?",
                  "confirm": "ଆପଣଙ୍କ ସମସ୍ତ ସୂଚନା ମିଳିଗଲା। ଅପଏଣ୍ଟମେଣ୍ଟ ବୁକ୍ କରିଦେବି କି?"},
}

# Slot field names vary by pack (doctor, stylist, table, room…). Only these are universal
# enough to have their own wording; everything else gets the generic "service" question.
_ASK_FIELD = {"name": "name", "full_name": "name", "patient": "name", "patient_name": "name",
              "phone": "phone", "mobile": "phone", "number": "phone", "contact": "phone",
              "datetime": "datetime", "date": "datetime", "time": "datetime",
              "when": "datetime", "slot": "datetime", "confirm": "confirm"}


def ask_line(field: str, lang_name: str | None, romanized: bool = False) -> str:
    """The pre-written question for a missing booking slot, in the caller's language."""
    key = _ASK_FIELD.get(str(field).strip().lower(), "service")
    if romanized:
        roman = SAFE_LINES_ROMAN.get(lang_name or "")
        if roman and f"ask_{key}" in roman:
            return roman[f"ask_{key}"]
        return ASK_LINES["English"][key]          # no romanised template → plain English
    return ASK_LINES.get(lang_name or "", ASK_LINES["English"])[key]


# --------------------------------------------------------------------------- #
# First-person echo — the model answering AS the caller
# --------------------------------------------------------------------------- #
# Observed repeatedly on real calls with qwen3:4b:
#   caller "मेरा नाम मनु मिश्रा है"        → agent "मेरा नाम मिश्रा है"
#   caller "8920429057"                    → agent "मेरा मोबाइल नंबर 8920429057 है"
#   caller "मैंने अपना नंबर आपको दे दिया है" → agent (verbatim)
# It also corrupted a real booking: #11 stored the name as "मिश्रा", losing "मनु".
#
# A receptionist speaks ABOUT the caller ("आपका नाम…"), never AS them ("मेरा नाम…").
# Grammatical person is what separates an echo from a legitimate confirmation, which
# necessarily repeats the caller's words back.
_FIRST_PERSON = frozenset("""
मेरा मेरी मेरे मुझे मैं मैंने अपना अपनी अपने
నా నాకు నేను మా
என் எனது எனக்கு நான்
ನನ್ನ ನನಗೆ ನಾನು
എന്റെ എനിക്ക് ഞാൻ
আমার আমাকে আমি
माझं माझा माझी मला मी
મારું મારો મારી મને હું
ਮੇਰਾ ਮੇਰੀ ਮੈਨੂੰ ਮੈਂ
میرا میری مجھے میں
ମୋର ମୋତେ ମୁଁ
my mine i've i'm myself
mera meri mere mujhe maine apna apni apne hamara hamari humara naa
naaku nenu naa
enaku naan
""".split())
# Second-person markers: their presence means the reply is addressed TO the caller, which
# is what a receptionist should sound like — so it is not an echo even if words overlap.
_SECOND_PERSON = frozenset("""
आपका आपकी आपके आपको आप तुम्हारा
మీ మీకు మీరు
உங்கள் உங்களுக்கு நீங்கள்
ನಿಮ್ಮ ನಿಮಗೆ ನೀವು
താങ്കളുടെ നിങ്ങളുടെ നിങ്ങൾ
আপনার আপনি
तुमचं तुमचा तुमची तुम्हाला तुम्ही
તમારું તમારો તમારી તમને તમે
ਤੁਹਾਡਾ ਤੁਹਾਨੂੰ ਤੁਸੀਂ
آپ آپکا
ଆପଣଙ୍କ ଆପଣ
your yours you
aapka aapki aapke aapko aap tumhara tumhe tum
mee meeku meeru
ungal ungalukku neengal
""".split())


def looks_like_echo(reply: str, caller: str, caller_values=()) -> bool:
    """True when the reply is the CALLER's own sentence played back in their voice.

    Two independent signatures:

      1. MIRRORED PRONOUN — the reply reuses a first-person pronoun the caller just
         used, and most of its words came from the caller's turn.
         ("मेरा नाम मनु मिश्रा है" → "मेरा नाम मिश्रा है")
         The overlap is matched fuzzily, because both sides come out of Whisper and
         "मोबाइल" arrives as "मोबाल" often enough to break exact comparison.

      2. CLAIMED POSSESSION — the reply speaks of a value the CALLER supplied as
         though it were its own. ("मेरा मोबाइल नंबर 8920429057 है")
         Here a second-person marker DOES clear the reply, since "आपका मोबाइल नंबर
         8920429057 सही है?" is a proper confirmation.

    Signature 1 deliberately does not veto on second-person markers: the caller's own
    sentence often contains one ("मैंने अपना नंबर आपको दे दिया है"), and echoing that
    verbatim is exactly the failure being caught.
    """
    r_words = _norm_words(reply)
    if len(r_words) < 3:
        return False
    r_set, c_set = set(r_words), set(_norm_words(caller))

    if r_set & c_set & _FIRST_PERSON:
        # Measure containment over the whole reply AND over its opening sentence. The
        # model often plays the caller's line back and then appends a real question
        # ("Mera naam Manu hai. Kya aapko phone number bhi chahiye?") — that tail
        # dilutes whole-reply overlap below any useful threshold, but leading with
        # their words in their voice is exactly the failure.
        best = _fuzzy_containment(r_set, c_set)
        head = _norm_words(re.split(r"[.?!।॥]", reply.strip())[0])
        if len(head) >= 3:
            best = max(best, _fuzzy_containment(set(head), c_set))
        if best >= 0.6:
            return True

    if (r_set & _FIRST_PERSON) and not (r_set & _SECOND_PERSON):
        reply_numbers = numbers_in(reply)
        if any(str(v) in reply_numbers for v in caller_values if str(v).isdigit()):
            return True
        # 3. WRONG POSSESSIVE — asking for the caller's detail in the first person:
        #    "मेरा मोबाइल नंबर क्या है?" is "what is MY mobile number?". It repeats
        #    nothing, so overlap cannot see it, but a receptionist asking for a detail
        #    must address the caller. Only POSSESSIVES count: "क्या मैं बुक कर दूँ?"
        #    ("shall I book it?") is a proper question about herself.
        # …and the possessive must be in the SAME sentence as the question. Our own
        # refusal line says "that detail is not with me" and, three clauses later,
        # "anything else?" — two separate thoughts, not a misused possessive.
        for part in re.split(r"[.!?।॥]+", reply):
            pw = set(_norm_words(part))
            if (pw & _FIRST_POSSESSIVE) and not (pw & _SECOND_PERSON) \
                    and _is_question(part + "?" if part is reply else part, pw):
                return True
    return False


# "my", not "I" — the subject pronoun is legitimate in a receptionist's own question.
_FIRST_POSSESSIVE = frozenset("""
मेरा मेरी मेरे अपना अपनी अपने
నా మా என் எனது ನನ್ನ എന്റെ আমার माझं माझा माझी મારું મારો મારી ਮੇਰਾ ਮੇਰੀ میرا میری ମୋର
my mine mera meri mere apna apni apne
""".split())
_WH = frozenset("""
क्या कौन कितना कितनी कैसे कहाँ क्यों कब
ఏమిటి ఎంత ఎలా ఎక్కడ ఎప్పుడు என்ன எவ்வளவு ಏನು ಎಷ್ಟು എന്ത് কি কত काय किती શું કેટલું ਕੀ کیا କଣ
what which how when where who kya kaun kitna kaise kahan kab
""".split())


def _is_question(text: str, words: set) -> bool:
    return (text or "").strip().endswith(("?", "؟")) or bool(words & _WH)


# --------------------------------------------------------------------------- #
# "Have they finished talking?" — read from the words, not the silence
# --------------------------------------------------------------------------- #
# Silence alone cannot tell a finished sentence from someone searching for a word, and
# a caller who pauses to recall a phone number is the single most common way a turn gets
# cut in half. Once the speculative transcript exists we can look at what they actually
# said: a sentence essentially never ends on a postposition, a conjunction or "my".
_DANGLING = frozenset("""
और या पर कि तो का की के को में से पे मेरा मेरी मेरे आपका आपकी आपके मुझे हमारा एक जो अगर क्योंकि लेकिन
మరియు నా మీ కి కు లో తో అని ఒక ఎందుకంటే కానీ
மற்றும் என் உங்கள் ஒரு ஏனென்றால் ஆனால்
ಮತ್ತು ನನ್ನ ನಿಮ್ಮ ಒಂದು ಆದರೆ
ഒരു എന്റെ താങ്കളുടെ പക്ഷേ
এবং আমার আপনার একটা কারণ কিন্তু
आणि माझा माझी तुमचा एक कारण पण
અને મારું તમારું એક કારણ પણ
ਅਤੇ ਮੇਰਾ ਤੁਹਾਡਾ ਇੱਕ ਕਿਉਂਕਿ ਪਰ
اور میرا آپ کا ایک کیونکہ لیکن
ଏବଂ ମୋର ଆପଣଙ୍କ ଗୋଟିଏ କାରଣ କିନ୍ତୁ
and or but my your the a an to for of in on with at from because if when
aur mera meri mere aapka mujhe ek kyunki lekin ka ki ke ko mein se
""".split())


# Nouns that ANNOUNCE a value rather than being one. "मेरा मोबाइल नंबर" is a complete
# noun phrase and a completely unfinished sentence — the number itself is still coming.
# This is the exact turn that was shipped without its digits on call #11.
_ANNOUNCES = frozenset("""
नंबर नम्बर नाम मोबाइल फोन
నంబర్ పేరు మొబైల్ ఫోన్
எண் பெயர் மொபைல்
ಸಂಖ್ಯೆ ಹೆಸರು ಮೊಬೈಲ್
നമ്പർ പേര് മൊബൈൽ
নম্বর নাম মোবাইল
नंबर नाव मोबाईल
નંબર નામ મોબાઇલ
ਨੰਬਰ ਨਾਂ ਮੋਬਾਈਲ
نمبر نام موبائل
ନମ୍ବର ନାମ ମୋବାଇଲ
number name mobile phone naam
""".split())


# Words that ARE the whole answer. A caller who says one of these has finished — there
# is no continuation that changes the meaning, which is what makes them safe to close on.
_TERMINAL_WORDS = frozenset("""
yes yeah yep no nope ok okay sure right correct done
haan han haa haan-ji ji jee nahi nahin theek thik acha achha bilkul
हाँ हां जी नहीं ना ठीक अच्छा बिल्कुल सही
అవును కాదు సరే ఒప్పుకుంటాను
ஆம் இல்லை சரி
ಹೌದು ಇಲ್ಲ ಸರಿ
അതെ ഇല്ല ശരി
হ্যাঁ না ঠিক
होय नाही ठीक
હા ના બરાબર
ਹਾਂ ਨਹੀਂ ਠੀਕ
ہاں نہیں ٹھیک
ହଁ ନା ଠିକ
""".split())


def looks_complete(text: str, expect_phone: bool = False) -> bool:
    """True only when nothing plausibly follows — so the turn can be closed EARLY.

    The mirror of `looks_incomplete`, and deliberately much stricter, because the two
    mistakes are not symmetrical. A false "incomplete" costs the caller a pause. A false
    "complete" CHOPS THEIR SENTENCE IN HALF, which is the failure the endpointer has
    already been tuned twice to avoid ("it breaks my long sentences").

    So this fires on exactly two shapes, both of which admit no continuation:

      1. a phone number that has reached full length, when a phone is what we asked for
         — the digits are done and nothing after them changes the number;
      2. a bare yes/no/ok, alone — "haan" is the entire answer, and waiting 1200ms for
         a word that is already finished is most of what makes a call feel slow.

    Everything else returns False and keeps the normal window. A name is NOT here
    ("Manu" then "Mishra"), nor is a time ("tomorrow" then "morning"), nor a sentence
    that merely ends in punctuation — Whisper punctuates mid-utterance constantly.
    """
    if not text:
        return False
    if looks_incomplete(text, expect_phone=expect_phone):
        return False                      # the two must never both be true

    words = _norm_words(text)
    if not words:
        return False

    # 1. a complete phone number, when a phone is what was asked for
    if expect_phone:
        digits = [n for n in numbers_in(text) if n.isdigit()]
        if digits and 10 <= len(max(digits, key=len)) <= 13:
            return True

    # 2. a bare affirmative or negative, and nothing else. Length is the guard: "yes"
    #    is finished, "yes I would like to book" is a sentence that may continue.
    if len(words) == 1 and words[0] in _TERMINAL_WORDS:
        return True
    if len(words) == 2 and all(w in _TERMINAL_WORDS for w in words):
        return True                       # "haan ji", "yes ok", "ठीक है" style pairs

    return False


def looks_incomplete(text: str, expect_phone: bool = False) -> bool:
    """True when the caller almost certainly has not finished the sentence.

    Two signals, both deliberately conservative — a false "incomplete" costs the caller
    a pause, but a false "complete" chops their sentence in half, which is the bug this
    exists to prevent:

      1. the last word is one a sentence does not end on ("मेरा", "and", "क्योंकि");
      2. a phone number is being read out and stops short of 10 digits
         ("मेरा मोबाइल नंबर 892" — the exact failure from call #11).
    """
    words = _norm_words(text)
    if not words:
        return False
    # A question is a finished utterance, whatever it ends on. Without this, English
    # "…anything else I can help you with?" and Marathi "…बुक करू का?" both read as
    # unfinished — "with" is a dangling preposition, and Marathi's interrogative "का"
    # is spelled exactly like Hindi's genitive postposition in the same script.
    if (text or "").strip().endswith(("?", "؟")):
        return False
    if len(words) >= 2 and words[-1] in _DANGLING:
        return True
    # "मेरा मोबाइल नंबर" / "my name" — the value they were about to give is still missing.
    if len(words) >= 2 and words[-1] in _ANNOUNCES and not numbers_in(text):
        return True
    # A phone number being read out that stops short of a real one.
    if expect_phone and (text or "").strip()[-1:].isdigit():
        tail = [n for n in numbers_in(text) if n.isdigit()]
        if tail and len(max(tail, key=len)) < 10:
            return True
    return False


def _fuzzy_containment(r_set, c_set) -> float:
    """Share of the reply's words that also came out of the caller's mouth, tolerating
    the small transcription differences Whisper produces between two takes."""
    if not r_set or not c_set:
        return 0.0
    hits = sum(1 for w in r_set
               if w in c_set or any(_bigram_sim(w, c) >= 0.6 for c in c_set))
    return hits / len(r_set)


def _bigram_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if min(len(a), len(b)) < 3 or abs(len(a) - len(b)) > 3:
        return 0.0
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return 2 * len(A & B) / (len(A) + len(B)) if A and B else 0.0


def _norm_words(text: str):
    """Words, lowercased, punctuation stripped. Never uses \\w — it drops the combining
    marks that Indian scripts are built from (see orchestrator._words)."""
    return [w for w in re.split(r"[\s,.!?;:।॥…\"'()\[\]-]+", (text or "").lower()) if w]


def safe_line(kind: str, lang_name: str | None, pack: dict, romanized: bool = False,
              ref: str | int = "") -> str:
    """The pre-written reply for `kind` ("unknown" | "scope" | "repeat") in the caller's language.

    Falls back to English for a language we have no template for — an accurate
    English sentence beats a garbled native one.
    """
    # A whitelist, because an unrecognised key must degrade to a sensible sentence
    # rather than raise. Anything added to SAFE_LINES has to be listed here too — a new
    # key that is not was silently answered with "I don't have that detail", which is
    # how "are you still there?" first went out as a refusal.
    kind = kind if kind in ("unknown", "scope", "repeat", "ref", "go_on",
                            "still_there", "goodbye", "noisy", "faint") else "unknown"
    # (booking questions live in ASK_LINES / ask_line — they are per-slot, not per-kind)
    table = None
    if lang_name:
        table = (SAFE_LINES_ROMAN if romanized else SAFE_LINES).get(lang_name)
        if table is None and romanized:
            table = SAFE_LINES.get("English")     # no romanised template → plain English
    table = table or SAFE_LINES["English"]
    biz = (pack.get("business") or {}).get("name") or "this business"
    return table[kind].format(biz=biz, ref=ref)


# --------------------------------------------------------------------------- #
# Hinglish — an Indian language typed/spoken in Latin letters
# --------------------------------------------------------------------------- #
# Deliberately words with no common English meaning, so an English sentence can't
# trip this. Two distinct markers are required before we call it Hinglish.
_HINGLISH_MARKERS = frozenset("""
kya kyu kyun kaise kahan kab kitna kitni kitne hai hain nahi nahin mujhe mera meri
mere aapka aapko aapki tumhara tumhare humara hamari chahiye karna karo kijiye
bataiye batao dijiye hoga hota hoti sakta sakti sakte bhai yaar acha achha theek
haan zaroorat zaruri paisa paise rupaye baje kal aaj subah shaam raat ghar naam
number bolo suniye milega lagega lagta karke wala wali abhi phir bhi
""".split())


def looks_hinglish(text: str) -> bool:
    """True when Latin-script text is really an Indian language (Hinglish).

    Typed input carries no Whisper language detection, so without this a caller
    writing "bhai neurologist ka kya rate hai?" would be answered from the English
    templates. Voice calls get the same answer from Whisper; this covers the rest.
    """
    words = re.findall(r"[a-z]+", (text or "").lower())
    if not words:
        return False
    return len(_HINGLISH_MARKERS.intersection(words)) >= 2


# --------------------------------------------------------------------------- #
# Degeneracy — a reply that has collapsed into a loop
# --------------------------------------------------------------------------- #
_SPLIT_RE = re.compile(r"[.!?…।॥؟\n]+")


def is_degenerate(text: str, *, min_words: int = 10, min_unique: float = 0.42) -> bool:
    """True when the model stopped answering and started repeating itself.

    Pushed into a language it handles poorly, a small model often gives up mid-reply
    and emits the same clause two or three times over. It scores as fluent to the
    model and as nonsense to the caller, and no fact-check catches it — there are no
    facts in it to check. Two signals:

      * a whole sentence of real length repeated verbatim, or
      * so few distinct words that the reply is mostly one phrase going round.
    """
    t = (text or "").strip()
    if not t:
        return False
    seen = set()
    for part in _SPLIT_RE.split(t):
        s = " ".join(part.split())
        if len(s.split()) < 4:            # short fragments repeat innocently ("Yes.", "Okay.")
            continue
        if s in seen:
            return True
        seen.add(s)
    words = t.split()
    # A loop doesn't always land on sentence boundaries — the model often repeats a
    # clause mid-sentence ("…వరకు ఉంటారు, శనివారం నుండి … వరకు ఉంటారు"). Any run of 5
    # words appearing twice is repetition no genuine one-or-two-sentence reply contains.
    if len(words) >= min_words:
        grams = set()
        for i in range(len(words) - 4):
            g = " ".join(words[i:i + 5])
            if g in grams:
                return True
            grams.add(g)
        if len(set(words)) / len(words) < min_unique:
            return True
    return False


# --------------------------------------------------------------------------- #
# Truncation — a reply cut off by the token budget
# --------------------------------------------------------------------------- #
_ENDERS = ".!?…।॥؟"


def trim_incomplete(text: str) -> str:
    """Drop a trailing fragment left behind when the model hit its token limit.

    ONLY call this when the model actually reported `finish_reason == "length"`.
    A missing full stop is NOT evidence of truncation — models leave the final
    period off constantly — and trimming on that guess silently swallows the last
    sentence of a perfectly good reply.
    """
    t = (text or "").strip()
    if not t or t[-1] in _ENDERS or t[-1] in '"\')':
        return t
    cut = max(t.rfind(c) for c in _ENDERS)
    if cut <= 0:
        return t                      # nothing complete to fall back to — keep as-is
    head = t[:cut + 1].strip()
    return head if len(head) >= 12 else t


# --------------------------------------------------------------------------- #
# The check itself
# --------------------------------------------------------------------------- #
class GuardConfig:
    """Tunables (config.yaml → `guard:`). Every check can be switched off."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.check_numbers = bool(cfg.get("check_numbers", True))
        self.check_script = bool(cfg.get("check_script", True))
        self.check_repetition = bool(cfg.get("check_repetition", True))
        self.trim_truncated = bool(cfg.get("trim_truncated", True))
        # Below this share of target-script letters the reply is treated as
        # "not in the caller's language at all". Deliberately low: we only want
        # to catch a wholesale fall-back to English, not a stray proper noun.
        self.min_script_share = float(cfg.get("min_script_share", 0.25))
        self.log_only = bool(cfg.get("log_only", False))   # observe without rewriting


def check_reply(text: str, *, pack: dict, allowed_numbers, lang_name: str | None,
                romanized: bool = False, gcfg: GuardConfig | None = None,
                check_language: bool = True, truncated: bool = False):
    """Verify one reply. Returns `(text, reason)` — `reason` is None when the reply
    passed, otherwise a short string naming what failed (for logs and metrics).

    A failed reply is replaced with the safe line for the caller's language unless
    the guard is in `log_only` mode.
    """
    gcfg = gcfg or GuardConfig()
    if not gcfg.enabled or not (text or "").strip():
        return text, None

    # Trim ONLY when the model itself said it ran out of tokens. Inferring truncation
    # from a missing full stop cut the last sentence off healthy replies.
    out = trim_incomplete(text) if (gcfg.trim_truncated and truncated) else text
    reason = "truncated" if out != text else None

    if gcfg.check_numbers:
        bad = ungrounded_numbers(out, allowed_numbers)
        if bad:
            reason = f"ungrounded_numbers={','.join(bad[:4])}"
            if not gcfg.log_only:
                return safe_line("unknown", lang_name, pack, romanized), reason

    if gcfg.check_repetition and is_degenerate(out):
        reason = "degenerate_repetition"
        if not gcfg.log_only:
            return safe_line("unknown", lang_name, pack, romanized), reason

    # `check_language` is off for booking confirmations: they read back the caller's own
    # name and details, which may legitimately be in Latin script.
    if check_language and gcfg.check_script and lang_name and not romanized:
        script = LANG_SCRIPT.get(lang_name)
        if script and script_share(out, script) < gcfg.min_script_share:
            reason = f"wrong_language(expected={lang_name})"
            if not gcfg.log_only:
                return safe_line("unknown", lang_name, pack, romanized), reason

    return out, reason
