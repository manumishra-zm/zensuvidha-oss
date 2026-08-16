"""The audio inspector's drawing decisions, run as the REAL page source.

The waveform is the only place in this project where a wrong answer is invisible:
a colour is not asserted by anything, so a segment attributed to the wrong speaker,
or a removed stretch drawn as kept, looks exactly like a correct picture. Everything
else the inspector says is words, which are at least checkable by reading them.

So these tests pull the functions out of `web/index.html` itself and run them in
Node against the payload `zensuvidha/server.py` actually sends. Copying the logic
into the test would pin the copy, not the page.

Skipped when node is not installed — it is a developer tool here, not a runtime
dependency, and the Python suite must still pass without it.

Run:  pytest -q tests/test_inspector.py
"""
import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "web", "index.html")
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

# The functions under test. Pulled by name so a rename fails loudly here rather than
# leaving a test that silently checks nothing.
WANTED = ["segAt", "keptAt", "spkIndex", "simOf", "drawTurnWave"]


def _extract(src: str, name: str) -> str:
    """Return `function name(...){...}`, matched by counting braces.

    A regex cannot do this: every one of these functions contains braces, and two of
    them contain a `}` inside a string.
    """
    start = src.index("function %s(" % name)
    depth, i, opened = 0, start, False
    while i < len(src):
        c = src[i]
        if c == "{":
            depth, opened = depth + 1, True
        elif c == "}":
            depth -= 1
            if opened and depth == 0:
                return src[start:i + 1]
        i += 1
    raise AssertionError("unbalanced braces in " + name)


def _run(js: str):
    src = open(INDEX, encoding="utf-8").read()
    body = "\n".join(_extract(src, n) for n in WANTED)
    prog = STUBS + body + "\n" + js
    out = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout or "null")


# A canvas that records what would have been painted, so the test can ask which
# colour landed on a given moment of the recording.
STUBS = r"""
const SPK_COLOURS=['#34d399','#f87171','#a78bfa','#fbbf24','#38bdf8'];
const NEUTRAL='#60a5fa';
const painted=[];            // {x, colour, alpha}
function fakeCanvas(W,H){
  const g={ fillStyle:'#000', strokeStyle:'#000', globalAlpha:1,
    setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){},
    fillRect(x,y,w,h){ painted.push({x, y, h, colour:g.fillStyle, alpha:g.globalAlpha}); },
    roundRect(x,y,w,h){ this._pending={x,y,h}; }, fill(){ if(this._pending)
      painted.push({...this._pending, colour:g.fillStyle, alpha:g.globalAlpha}); },
    fillText(){}, font:'', textAlign:'' };
  return { clientWidth:W, clientHeight:H, width:0, height:0, getContext:()=>g };
}
global.window={devicePixelRatio:1};
global.getComputedStyle=()=>({getPropertyValue:k=>k==='--muted'?'#9aa2ad':'#eeeeee'});
global.document={documentElement:{}, getElementById:()=>null};
// Ask what colour the trace is at a given second — ignoring the backdrop fills, which
// are painted full-height at y=0 before the waveform itself.
// The trace is drawn as BARS WITH GAPS now, so an exact-column probe lands between
// them. Take the nearest painted bar instead — the question is what colour that moment
// is, not which pixel it starts on.
function colourAt(clip,t){
  const W=200, x=t/clip.seconds*W;
  const bars=painted.filter(p=>p.h<100);   // 100 = canvas height: backdrops are full-height
  if(!bars.length) return null;
  let best=bars[0];
  for(const p of bars) if(Math.abs(p.x-x) < Math.abs(best.x-x)) best=p;
  return best;
}
function draw(clip){ painted.length=0; drawTurnWave(fakeCanvas(200,100), clip); }
function tone(seconds){ const a=new Float32Array(Math.round(seconds*16000));
  for(let i=0;i<a.length;i++) a[i]=Math.sin(i/8)*0.5; return a; }
"""


def test_the_caller_is_green_whoever_spoke_first():
    """Colour is assigned by voiceprint match, not by segment order. If it followed
    order, a stranger opening the clip would be painted as the caller — and the whole
    point of the picture is showing which of the two you are looking at."""
    # The caller is speaker 2 and the stranger is speaker 0, deliberately: with the
    # cluster ids used as colour indices directly the caller would still come out
    # green whenever they happened to be cluster 0, and this test would pass on a
    # broken page. Proven by reverting — colouring by `spk` fails only this shape.
    out = _run(r"""
    const ins={segments:[{s:0,e:2,spk:0,sim:0.21,keep:false},
                         {s:2,e:5,spk:2,sim:0.88,keep:true}],
               kept_spans:[[2,5]]};
    const clip={pcm:tone(5), seconds:5, insight:ins};
    draw(clip);
    console.log(JSON.stringify({
      stranger: colourAt(clip,1.0), caller: colourAt(clip,3.5),
      callerIsFirstColour: colourAt(clip,3.5).colour===SPK_COLOURS[0]}));
    """)
    assert out["callerIsFirstColour"], out
    assert out["stranger"]["colour"] != out["caller"]["colour"]


def test_removed_audio_is_drawn_faint_and_kept_audio_is_not():
    """The one thing a reader takes from this picture at a glance."""
    out = _run(r"""
    const ins={segments:[{s:0,e:2,spk:0,sim:0.9,keep:true},
                         {s:2,e:4,spk:1,sim:0.2,keep:false}],
               kept_spans:[[0,2]]};
    const clip={pcm:tone(4), seconds:4, insight:ins};
    draw(clip);
    console.log(JSON.stringify({kept:colourAt(clip,1.0).alpha,
                                removed:colourAt(clip,3.0).alpha}));
    """)
    assert out["kept"] == 1
    assert out["removed"] < 1, "removed audio is drawn as solid as the caller's own"


def test_nothing_removed_means_nothing_is_faded():
    """kept_spans is None on every fail-open path. Fading anything there would tell the
    operator audio was filtered at the exact moment none was."""
    out = _run(r"""
    const ins={segments:[{s:0,e:3,spk:0,sim:null,keep:true}], kept_spans:null};
    const clip={pcm:tone(3), seconds:3, insight:ins};
    draw(clip);
    console.log(JSON.stringify({alphas:[...new Set(painted.map(p=>p.alpha))]}));
    """)
    assert out["alphas"] == [1], out


def test_a_turn_with_no_analysis_still_draws():
    """Speculative-shortcut turns never get an insight. The page must show the
    recording rather than an empty box, or those turns look like a failure."""
    out = _run(r"""
    const clip={pcm:tone(2), seconds:2, insight:null};
    draw(clip);
    console.log(JSON.stringify({bars:painted.length>20,
                                neutral:colourAt(clip,1.0).colour===NEUTRAL}));
    """)
    assert out["bars"] and out["neutral"], out


def test_a_cut_inside_a_segment_is_drawn_where_it_happened():
    """The second-pass rescan cuts INSIDE a segment. Colouring by the segment's `keep`
    flag alone would shade the whole segment one way and hide the cut entirely."""
    out = _run(r"""
    const ins={segments:[{s:0,e:6,spk:0,sim:0.8,keep:true}], kept_spans:[[0,2],[4,6]]};
    const clip={pcm:tone(6), seconds:6, insight:ins};
    draw(clip);
    console.log(JSON.stringify({inside:colourAt(clip,3.0).alpha,
                                before:colourAt(clip,1.0).alpha,
                                after:colourAt(clip,5.0).alpha}));
    """)
    assert out["before"] == 1 and out["after"] == 1
    assert out["inside"] < 1, "the cut in the middle of the segment is invisible"


def test_the_click_target_matches_the_segment_that_was_drawn():
    """Clicking plays a range and names a speaker. If segAt disagreed with what was
    painted, the colour under the cursor would not be the thing that plays."""
    out = _run(r"""
    const ins={segments:[{s:0,e:2,spk:1,sim:0.2,keep:false},
                         {s:2,e:5,spk:0,sim:0.88,keep:true}], kept_spans:[[2,5]]};
    const clip={pcm:tone(5), seconds:5, insight:ins};
    draw(clip);
    const probes=[0.5,1.9,2.1,4.9].map(t=>({
      t, seg:segAt(ins,t), drawn:colourAt(clip,t).colour,
      expect:SPK_COLOURS[spkIndex(ins,segAt(ins,t).spk)%SPK_COLOURS.length]}));
    console.log(JSON.stringify({ok:probes.every(p=>p.drawn===p.expect), probes}));
    """)
    assert out["ok"], out["probes"]


# --------------------------------------------------------------------------- #
# the endpoint window, as the page computes it
#
# Turn-taking now fuses four signals — how long they have been talking, how this caller
# pauses, what the words say, and what the pitch says. The failure mode is not that one
# is wrong; it is that a weaker signal overrides a stronger one. That ordering only
# exists in this function, so it is pinned here rather than reasoned about.
# --------------------------------------------------------------------------- #
ENDPOINT_FNS = ["endpointMs"]

ENDPOINT_STUBS = r"""
let uttMs=0, speakerPauseMs=0, holdForMore=false, settled=false, expectSlot=null,
    tone=null, fillerHold=false;
// The policy the server sends (zensuvidha/turn.py, "normal"). Kept in sync by
// test_the_client_and_the_server_agree_on_the_ladder below.
let TURN={base_ms:800, long_ms:1200, long_utt_ms:1500, max_ms:2000, settled_ms:400,
          hold_extra_ms:900, filler_extra_ms:500, pause_factor:1.25,
          expect_extra_ms:{phone:600, datetime:300, name:150},
          tone_scale:{finished:0.80, holding:1.25, unsure:1}, eagerness:'normal'};
function set(o){ uttMs=o.uttMs??0; speakerPauseMs=o.pause??0; holdForMore=!!o.hold;
                 settled=!!o.settled; expectSlot=o.expect??null; tone=o.tone??null;
                 fillerHold=!!o.filler; if(o.policy) TURN=o.policy; }
"""


def _run_endpoint(js: str):
    src = open(INDEX, encoding="utf-8").read()
    body = "\n".join(_extract(src, n) for n in ENDPOINT_FNS)
    prog = ENDPOINT_STUBS + body + "\n" + js
    out = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout or "null")


def test_the_words_outrank_the_voice():
    """"my mobile number is", said with a textbook falling contour, has still not given
    us the number. A caller reading digits pauses between groups, and this is precisely
    the turn that must not be closed early."""
    out = _run_endpoint(r"""
    set({uttMs:2000, hold:true, tone:'finished', expect:'phone'});
    const held = endpointMs();
    set({uttMs:2000, hold:true, tone:null, expect:'phone'});
    console.log(JSON.stringify({held, plain: endpointMs()}));
    """)
    assert out["held"] == out["plain"], out


def test_a_finished_contour_shortens_an_ordinary_turn():
    out = _run_endpoint(r"""
    set({uttMs:2000}); const plain = endpointMs();
    set({uttMs:2000, tone:'finished'}); const fin = endpointMs();
    set({uttMs:2000, tone:'holding'}); const hold = endpointMs();
    set({uttMs:2000, tone:'unsure'}); const unsure = endpointMs();
    console.log(JSON.stringify({plain, fin, hold, unsure}));
    """)
    assert out["fin"] < out["plain"] < out["hold"]
    assert out["unsure"] == out["plain"], "an unsure reading changed the window"


def test_the_voice_never_becomes_the_decider():
    """It is a third opinion. However confident it is, it may not move the window
    outside what the other signals already allowed."""
    out = _run_endpoint(r"""
    const rows=[];
    for(const t of [null,'finished','holding','unsure'])
      for(const u of [500, 2000, 6000]){ set({uttMs:u, tone:t}); rows.push(endpointMs()); }
    console.log(JSON.stringify({min:Math.min(...rows), max:Math.max(...rows)}));
    """)
    assert out["min"] >= 400, out
    assert out["max"] <= 2000 * 1.25 + 1, out


# --------------------------------------------------------------------------- #
# `hidden` has to actually hide
# --------------------------------------------------------------------------- #
def test_every_hidden_overlay_has_a_rule_that_hides_it():
    """An author `display:` BEATS the browser's own [hidden]{display:none}.

    So the moment a class sets `display:flex`, the `hidden` attribute silently stops
    working: the element is visible from page load, covering whatever is behind it, and
    the JS that sets `.hidden = true` to close it appears to do nothing — because it is
    changing an attribute no rule is listening to.

    This shipped exactly once, on the audio inspector. `.voice-fs[hidden]` had been
    written for the same reason years earlier, which is what made the omission easy to
    miss: the pattern was right there and half-copied. Checked for every overlay rather
    than for that one, because the next full-screen panel will hit it too.
    """
    import re
    src = open(INDEX, encoding="utf-8").read()

    # classes used on an element that carries the `hidden` attribute in the markup
    hidden_classes = set()
    for tag in re.findall(r"<div\b[^>]*\bhidden\b[^>]*>", src):
        m = re.search(r'class="([^"]+)"', tag)
        if m:
            hidden_classes.update(m.group(1).split())

    assert hidden_classes, "no hidden overlays found — has the markup changed?"
    for cls in sorted(hidden_classes):
        # …only matters when a rule for that class sets `display`
        rules = re.findall(r"\.%s\{([^}]*)\}" % re.escape(cls), src)
        if not any("display:" in r for r in rules):
            continue
        assert re.search(r"\.%s\[hidden\]\s*\{[^}]*display\s*:\s*none" % re.escape(cls), src), (
            "'.%s' sets display, so the hidden attribute cannot hide it — "
            "add '.%s[hidden]{display:none;}'" % (cls, cls))


def test_a_canvas_with_no_size_is_retried_rather_than_drawn_into():
    """A canvas that has not been laid out yet reports clientWidth 0, and every loop in
    drawWave then runs zero times — painting nothing, throwing nothing, and leaving a
    blank box that looks exactly like a broken feature. This is drawn one frame after
    the panel is revealed, and one frame is not always enough."""
    out = _run(r"""
    let scheduled=0;
    global.requestAnimationFrame=(fn)=>{ scheduled++; if(scheduled<3) fn(); };
    const clip={pcm:tone(2), seconds:2, insight:null};
    const g={fillStyle:'',strokeStyle:'',globalAlpha:1,setTransform(){},clearRect(){},
             beginPath(){},moveTo(){},lineTo(){},stroke(){},fillRect(){painted.push({});},
             fillText(){},font:'',textAlign:''};
    const dead={clientWidth:0, clientHeight:0, width:0, height:0, getContext:()=>g};
    painted.length=0;
    drawTurnWave(dead, clip);
    console.log(JSON.stringify({retried: scheduled>0, painted: painted.length}));
    """)
    assert out["retried"], "a zero-size canvas was silently drawn into nothing"
    assert out["painted"] == 0


def test_a_turn_with_no_audio_says_so_on_the_canvas():
    """A silent blank box is indistinguishable from a bug, and this is the one panel
    whose whole job is showing what happened."""
    out = _run(r"""
    const texts=[];
    const g={fillStyle:'',strokeStyle:'',globalAlpha:1,setTransform(){},clearRect(){},
             beginPath(){},moveTo(){},lineTo(){},stroke(){},fillRect(){},
             fillText(t){texts.push(t);},font:'',textAlign:''};
    const cv={clientWidth:400, clientHeight:200, width:0, height:0, getContext:()=>g};
    drawTurnWave(cv, {pcm:new Float32Array(0), seconds:0, insight:null});
    console.log(JSON.stringify({texts}));
    """)
    assert any("no audio" in t for t in out["texts"]), out


def test_no_two_top_level_functions_share_a_name():
    """THE ONE THAT COST FOUR ROUNDS OF GUESSING.

    The inspector's waveform renderer was called `drawWave`. So was the orb's mic-level
    renderer, three hundred lines below. Function declarations hoist and the LAST one
    wins, so every `drawWave(cv, clip)` in the inspector silently invoked the orb's
    `drawWave(A)` with a canvas element — painting nothing, throwing nothing, leaving a
    blank box.

    It survived four fixes because the tests in this file EXTRACT BY NAME and find the
    first definition, so they exercised the right function while the browser ran the
    wrong one. Green tests, blank UI. Only rendering the page in a real browser found
    it. A duplicate name is cheap to detect and expensive to chase.
    """
    import re
    from collections import Counter
    src = open(INDEX, encoding="utf-8").read()
    # top-level declarations only — nested helpers legitimately reuse short names
    names = re.findall(r"^function ([A-Za-z_$][\w$]*)\s*\(", src, re.M)
    dupes = {n: c for n, c in Counter(names).items() if c > 1}
    assert not dupes, (
        "two top-level functions share a name — the later one silently wins for every "
        "call: %s" % dupes)
