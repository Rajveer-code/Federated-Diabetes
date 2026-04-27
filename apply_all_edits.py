"""
apply_all_edits.py
==================
Applies all 13 manuscript edits (4A–4M) to manuscript_v4_unpacked/word/document.xml
Run from D:/Projects/diabetes_prediction_project/
"""

import os, sys, re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DOC = os.path.join('manuscript_v4_unpacked', 'word', 'document.xml')

with open(DOC, 'rb') as f:
    raw = f.read()
content = raw.decode('utf-8', errors='replace')

original_len = len(content)
edits_applied = []

# ─────────────────────────────────────────────────────────────────────────────
# XML BUILDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

BORDERS = """\
            <w:tcBorders>
              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>
              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>
              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>
              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>
            </w:tcBorders>"""

TNRFONTS = '<w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'

def tc_data(width, text, align='center'):
    return f"""\
        <w:tc>
          <w:tcPr>
            <w:tcW w:type="dxa" w:w="{width}"/>
{BORDERS}
            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>
          </w:tcPr>
          <w:p>
            <w:pPr>
              <w:spacing w:after="60" w:before="60"/>
              <w:jc w:val="{align}"/>
            </w:pPr>
            <w:r>
              <w:rPr>
                {TNRFONTS}
                <w:b w:val="false"/>
                <w:bCs w:val="false"/>
                <w:sz w:val="20"/>
                <w:szCs w:val="20"/>
              </w:rPr>
              <w:t xml:space="preserve">{text}</w:t>
            </w:r>
          </w:p>
        </w:tc>"""

def tc_hdr(width, text, align='center'):
    return f"""\
        <w:tc>
          <w:tcPr>
            <w:tcW w:type="dxa" w:w="{width}"/>
{BORDERS}
            <w:shd w:fill="D5E8F0" w:color="auto" w:val="clear"/>
          </w:tcPr>
          <w:p>
            <w:pPr>
              <w:spacing w:after="60" w:before="60"/>
              <w:jc w:val="{align}"/>
            </w:pPr>
            <w:r>
              <w:rPr>
                {TNRFONTS}
                <w:b/>
                <w:bCs/>
                <w:sz w:val="20"/>
                <w:szCs w:val="20"/>
              </w:rPr>
              <w:t xml:space="preserve">{text}</w:t>
            </w:r>
          </w:p>
        </w:tc>"""

def hdr_row(*cells):
    inner = '\n'.join(cells)
    return f"""\
      <w:tr>
        <w:trPr>
          <w:tblHeader/>
        </w:trPr>
{inner}
      </w:tr>"""

def data_row(*cells):
    inner = '\n'.join(cells)
    return f"""\
      <w:tr>
{inner}
      </w:tr>"""

def tbl(col_widths, *rows):
    grid = '\n'.join(f'        <w:gridCol w:w="{w}"/>' for w in col_widths)
    rows_xml = '\n'.join(rows)
    return f"""\
    <w:tbl>
      <w:tblPr>
        <w:tblW w:type="dxa" w:w="9360"/>
        <w:tblBorders>
          <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>
          <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>
          <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>
          <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>
          <w:insideH w:val="single" w:color="CCCCCC" w:sz="1"/>
          <w:insideV w:val="single" w:color="CCCCCC" w:sz="1"/>
        </w:tblBorders>
      </w:tblPr>
      <w:tblGrid>
{grid}
      </w:tblGrid>
{rows_xml}
    </w:tbl>"""

def p_body(text):
    return f"""\
    <w:p>
      <w:pPr>
        <w:spacing w:after="120" w:before="0" w:line="480" w:lineRule="auto"/>
        <w:jc w:val="both"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          {TNRFONTS}
          <w:b w:val="false"/>
          <w:bCs w:val="false"/>
          <w:i w:val="false"/>
          <w:iCs w:val="false"/>
          <w:sz w:val="24"/>
          <w:szCs w:val="24"/>
        </w:rPr>
        <w:t xml:space="preserve">{text}</w:t>
      </w:r>
    </w:p>"""

def p_heading_sub(text):
    """Sub-section heading like '4.4 Fairness Analysis'."""
    return f"""\
    <w:p>
      <w:pPr>
        <w:spacing w:after="80" w:before="200"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          {TNRFONTS}
          <w:b/>
          <w:bCs/>
          <w:i w:val="false"/>
          <w:iCs w:val="false"/>
          <w:sz w:val="24"/>
          <w:szCs w:val="24"/>
        </w:rPr>
        <w:t xml:space="preserve">{text}</w:t>
      </w:r>
    </w:p>"""

def p_table_title(text):
    """Bold table title like 'Table 6. ...'"""
    return f"""\
    <w:p>
      <w:pPr>
        <w:spacing w:after="80" w:before="240"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          {TNRFONTS}
          <w:b/>
          <w:bCs/>
          <w:sz w:val="22"/>
          <w:szCs w:val="22"/>
        </w:rPr>
        <w:t xml:space="preserve">{text}</w:t>
      </w:r>
    </w:p>"""

def p_blank():
    return f"""\
    <w:p>
      <w:pPr>
        <w:spacing w:line="480" w:lineRule="auto"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          {TNRFONTS}
          <w:sz w:val="24"/>
          <w:szCs w:val="24"/>
        </w:rPr>
        <w:t xml:space="preserve"/>
      </w:r>
    </w:p>"""

def p_fig_caption(text):
    return f"""\
    <w:p>
      <w:pPr>
        <w:spacing w:after="160" w:before="80"/>
        <w:jc w:val="center"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          {TNRFONTS}
          <w:i/>
          <w:iCs/>
          <w:sz w:val="20"/>
          <w:szCs w:val="20"/>
        </w:rPr>
        <w:t xml:space="preserve">{text}</w:t>
      </w:r>
    </w:p>"""

def apply(old, new, label):
    global content
    if old not in content:
        print(f"  WARN [{label}]: anchor not found!")
        return False
    count = content.count(old)
    if count > 1:
        print(f"  WARN [{label}]: anchor appears {count} times, replacing first only")
        content = content.replace(old, new, 1)
    else:
        content = content.replace(old, new)
    edits_applied.append(label)
    print(f"  OK  [{label}]")
    return True

def apply_all(old, new, label):
    global content
    if old not in content:
        print(f"  WARN [{label}]: anchor not found!")
        return False
    count = content.count(old)
    content = content.replace(old, new)
    edits_applied.append(label)
    print(f"  OK  [{label}] ({count} replacements)")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4A: Centralised NN row in Table 2 (NHANES internal)
# Insert before </w:tbl> that follows FedNova row ending with Spec=0.651
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4A] Inserting Centralised NN row in Table 2...")

# Exp1 internal: AUC=0.801 [0.782–0.819], Brier=0.187, F1=0.494, Sens=0.794, Spec=0.676
cnn_t2 = data_row(
    tc_data(2200, 'Centralised NN (DiabetesNet)', 'left'),
    tc_data(1300, '0.801'),
    tc_data(1800, '0.782\u20130.819'),
    tc_data(900,  '0.187'),
    tc_data(900,  '0.494'),
    tc_data(1080, '0.794'),
    tc_data(1180, '0.676'),
)

# Unique anchor: last cell of Table 2 FedNova row (Spec=0.651), w:w="1180"
anchor_4a = (
    '              <w:t xml:space="preserve">0.651</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    '    </w:tbl>'
)
replace_4a = (
    '              <w:t xml:space="preserve">0.651</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    + cnn_t2 + '\n'
    '    </w:tbl>'
)
apply(anchor_4a, replace_4a, '4A')


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4B: Centralised NN row in Table 3 (BRFSS external)
# Insert before </w:tbl> that follows FedNova row ending with Spec=0.635
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4B] Inserting Centralised NN row in Table 3...")

# Exp1 external: AUC=0.749 [0.748–0.750], Brier=0.306, F1=0.356, Sens=0.720, Spec=0.643
cnn_t3 = data_row(
    tc_data(2200, 'Centralised NN (DiabetesNet)', 'left'),
    tc_data(1200, '0.749'),
    tc_data(1760, '0.748\u20130.750'),
    tc_data(900,  '0.306'),
    tc_data(900,  '0.356'),
    tc_data(1100, '0.720'),
    tc_data(1300, '0.643'),
)

# Unique anchor: last cell of Table 3 FedNova row (Spec=0.635), w:w="1300"
anchor_4b = (
    '              <w:t xml:space="preserve">0.635</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    '    </w:tbl>'
)
replace_4b = (
    '              <w:t xml:space="preserve">0.635</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    + cnn_t3 + '\n'
    '    </w:tbl>'
)
apply(anchor_4b, replace_4b, '4B')


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4C: Fix Table 4 — rename Source→Domain, add Centralised NN row,
#          fix Figure 11 caption, fix Section 4.4 text
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4C] Fixing Table 4 and related text...")

# 4C-i: Rename "Source" header cell → "Domain"
apply(
    '              <w:t xml:space="preserve">Source</w:t>',
    '              <w:t xml:space="preserve">Domain</w:t>',
    '4C-i (Source→Domain header)'
)

# 4C-ii: Add Centralised NN row in Table 4 after Centralised XGBoost row
# Table 4 cols: model(2100), auc1839(1380), auc60(1380), gap(1100), gapvs(1500), domain(1900)
# Exp1 external subgroups: age18-39=0.720, age60+=0.657, gap=0.063
# Gap vs Centralised: (0.063-0.069)/0.069 × 100 = -8.7%
cnn_t4 = data_row(
    tc_data(2100, 'Centralised NN (DiabetesNet)', 'left'),
    tc_data(1380, '0.720'),
    tc_data(1380, '0.657'),
    tc_data(1100, '0.063'),
    tc_data(1500, '\u22128.7%'),
    tc_data(1900, 'BRFSS external'),
)

# Unique anchor: last cell of Centralised XGBoost row in Table 4 (BRFSS external, after 0.069 row)
# The Centralised XGBoost row has "BRFSS external" in Source at L6192
# Its preceding cells have: 0.656, 0.587, 0.069, Baseline
anchor_4c_ii = (
    '              <w:t xml:space="preserve">Baseline</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="1900"/>\n'
    '            <w:tcBorders>\n'
    '              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '            </w:tcBorders>\n'
    '            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>\n'
    '          </w:tcPr>\n'
    '          <w:p>\n'
    '            <w:pPr>\n'
    '              <w:spacing w:after="60" w:before="60"/>\n'
    '              <w:jc w:val="center"/>\n'
    '            </w:pPr>\n'
    '            <w:r>\n'
    '              <w:rPr>\n'
    '                <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '                <w:b w:val="false"/>\n'
    '                <w:bCs w:val="false"/>\n'
    '                <w:sz w:val="20"/>\n'
    '                <w:szCs w:val="20"/>\n'
    '              </w:rPr>\n'
    '              <w:t xml:space="preserve">BRFSS external</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    '      <w:tr>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="2100"/>\n'
    '            <w:tcBorders>\n'
    '              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '            </w:tcBorders>\n'
    '            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>\n'
    '          </w:tcPr>\n'
    '          <w:p>\n'
    '            <w:pPr>\n'
    '              <w:spacing w:after="60" w:before="60"/>\n'
    '              <w:jc w:val="left"/>\n'
    '            </w:pPr>\n'
    '            <w:r>\n'
    '              <w:rPr>\n'
    '                <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '                <w:b w:val="false"/>\n'
    '                <w:bCs w:val="false"/>\n'
    '                <w:sz w:val="20"/>\n'
    '                <w:szCs w:val="20"/>\n'
    '              </w:rPr>\n'
    '              <w:t xml:space="preserve">FedAvg (ext.)</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>'
)

# This is complex. Let's use a simpler anchor: after the Centralised XGBoost row ends
# The Centralised XGBoost row has "BRFSS external" at w:w="1900" followed by </w:tr>
# then FedAvg row starts. Insert the new row between them.
# Unique: after "BRFSS external" (from XGBoost row which has "Baseline" before "BRFSS external")
# Use "Baseline" + close + "BRFSS external" + row close + FedAvg row start
anchor_4c_ii = (
    '              <w:t xml:space="preserve">Baseline</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="1900"/>'
)

replace_4c_ii_check = anchor_4c_ii  # just checking existence

# Simpler: the Centralised XGBoost row in Table 4 is the one with "Baseline" in gap-vs-centralised
# and "BRFSS external" in domain. Let me anchor on closing that row.
# After seeing "BRFSS external" and "</w:tr>" which closes the Centralised XGBoost row
# then the FedAvg row opens with FedAvg (ext.)

# Most reliable: Replace the string ending the Centralised XGBoost row in Table 4
# The row ends: ... "BRFSS external" </w:t>...</w:tr> then <w:tr> (FedAvg starts)
# Use the Baseline cell + domain cell combination
apply(
    '              <w:t xml:space="preserve">Baseline</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="1900"/>\n'
    '            <w:tcBorders>\n'
    '              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '            </w:tcBorders>\n'
    '            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>\n'
    '          </w:tcPr>\n'
    '          <w:p>\n'
    '            <w:pPr>\n'
    '              <w:spacing w:after="60" w:before="60"/>\n'
    '              <w:jc w:val="center"/>\n'
    '            </w:pPr>\n'
    '            <w:r>\n'
    '              <w:rPr>\n'
    '                <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '                <w:b w:val="false"/>\n'
    '                <w:bCs w:val="false"/>\n'
    '                <w:sz w:val="20"/>\n'
    '                <w:szCs w:val="20"/>\n'
    '              </w:rPr>\n'
    '              <w:t xml:space="preserve">BRFSS external</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    '      <w:tr>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="2100"/>\n'
    '            <w:tcBorders>\n'
    '              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '            </w:tcBorders>\n'
    '            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>\n'
    '          </w:tcPr>\n'
    '          <w:p>\n'
    '            <w:pPr>\n'
    '              <w:spacing w:after="60" w:before="60"/>\n'
    '              <w:jc w:val="left"/>\n'
    '            </w:pPr>\n'
    '            <w:r>\n'
    '              <w:rPr>\n'
    '                <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '                <w:b w:val="false"/>\n'
    '                <w:bCs w:val="false"/>\n'
    '                <w:sz w:val="20"/>\n'
    '                <w:szCs w:val="20"/>\n'
    '              </w:rPr>\n'
    '              <w:t xml:space="preserve">FedAvg (ext.)</w:t>',
    # REPLACEMENT:
    '              <w:t xml:space="preserve">Baseline</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="1900"/>\n'
    '            <w:tcBorders>\n'
    '              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '            </w:tcBorders>\n'
    '            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>\n'
    '          </w:tcPr>\n'
    '          <w:p>\n'
    '            <w:pPr>\n'
    '              <w:spacing w:after="60" w:before="60"/>\n'
    '              <w:jc w:val="center"/>\n'
    '            </w:pPr>\n'
    '            <w:r>\n'
    '              <w:rPr>\n'
    '                <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '                <w:b w:val="false"/>\n'
    '                <w:bCs w:val="false"/>\n'
    '                <w:sz w:val="20"/>\n'
    '                <w:szCs w:val="20"/>\n'
    '              </w:rPr>\n'
    '              <w:t xml:space="preserve">BRFSS external</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    + cnn_t4 + '\n'
    '      <w:tr>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="2100"/>\n'
    '            <w:tcBorders>\n'
    '              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
    '            </w:tcBorders>\n'
    '            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>\n'
    '          </w:tcPr>\n'
    '          <w:p>\n'
    '            <w:pPr>\n'
    '              <w:spacing w:after="60" w:before="60"/>\n'
    '              <w:jc w:val="left"/>\n'
    '            </w:pPr>\n'
    '            <w:r>\n'
    '              <w:rPr>\n'
    '                <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '                <w:b w:val="false"/>\n'
    '                <w:bCs w:val="false"/>\n'
    '                <w:sz w:val="20"/>\n'
    '                <w:szCs w:val="20"/>\n'
    '              </w:rPr>\n'
    '              <w:t xml:space="preserve">FedAvg (ext.)</w:t>',
    '4C-ii (Table 4 Centralised NN row)'
)

# 4C-iii: Fix Figure 11 caption — remove "and 60% from the published internal gap (0.135)"
apply(
    'representing a 21.7% reduction from the centralised baseline (0.069) and 60% from the published internal gap (0.135). All FL models improve fairness relative to the centralised replication.',
    'representing a 21.7% reduction from the centralised XGBoost external baseline (0.069). All FL models improve fairness relative to the centralised replication.',
    '4C-iii (Fig 11 caption fix)'
)

# 4C-iv: Fix Section 4.4 text — same fix
apply(
    'representing a 21.7% reduction from the centralised baseline (0.069) and a 60% reduction from the published internal gap (0.135).',
    'representing a 21.7% reduction from the centralised XGBoost external baseline (0.069).',
    '4C-iv (Sec 4.4 text fix)'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4D+4E: New Tables 6 and 7 after Table 5 caption area
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4D+4E] Inserting Tables 6 (Node B Ablation) and 7 (FedProx sensitivity)...")

# Table 6 cols: Config(3000), Ext AUC [95% CI](2400), AUC >=60(1300), AUC 18-39(1300), Gap(1360)
# = 9360 total
t6_rows = [
    hdr_row(
        tc_hdr(3000, 'Configuration', 'left'),
        tc_hdr(2400, 'External AUC [95% CI]'),
        tc_hdr(1300, 'AUC (\u226560)'),
        tc_hdr(1300, 'AUC (18\u201339)'),
        tc_hdr(1360, 'Elderly Gap'),
    ),
    data_row(
        tc_data(3000, 'FedAvg (Full: A+B+C)', 'left'),
        tc_data(2400, '0.757 [0.756\u20130.758]'),
        tc_data(1300, '0.669'),
        tc_data(1300, '0.722'),
        tc_data(1360, '0.054'),
    ),
    data_row(
        tc_data(3000, 'FedAvg Ablation (A+C only, Node B removed)', 'left'),
        tc_data(2400, '0.739 [0.738\u20130.740]'),
        tc_data(1300, '0.648'),
        tc_data(1300, '0.718'),
        tc_data(1360, '0.070 (+29.1%)'),
    ),
    data_row(
        tc_data(3000, 'Centralised NN (DiabetesNet, baseline)', 'left'),
        tc_data(2400, '0.749 [0.748\u20130.750]'),
        tc_data(1300, '0.657'),
        tc_data(1300, '0.720'),
        tc_data(1360, '0.063'),
    ),
    data_row(
        tc_data(3000, 'Centralised XGBoost (ext.)', 'left'),
        tc_data(2400, '0.700 [N/A]'),
        tc_data(1300, '0.587'),
        tc_data(1300, '0.656'),
        tc_data(1360, '0.069'),
    ),
]
table6_xml = tbl([3000, 2400, 1300, 1300, 1360], *t6_rows)

# Table 7 cols: mu(900), AUC(1400), CI(2000), Elderly AUC(1800), Young AUC(1800), Gap(1460) = 9360
t7_rows = [
    hdr_row(
        tc_hdr(900, '\u03bc', 'left'),
        tc_hdr(1400, 'Ext AUC'),
        tc_hdr(2000, '95% CI'),
        tc_hdr(1800, 'AUC (\u226560)'),
        tc_hdr(1800, 'AUC (18\u201339)'),
        tc_hdr(1460, 'Elderly Gap'),
    ),
    data_row(
        tc_data(900, '0.01', 'left'),
        tc_data(1400, '0.747'),
        tc_data(2000, '0.746\u20130.748'),
        tc_data(1800, '0.656'),
        tc_data(1800, '0.713'),
        tc_data(1460, '0.057'),
    ),
    data_row(
        tc_data(900, '0.05', 'left'),
        tc_data(1400, '0.755'),
        tc_data(2000, '0.754\u20130.756'),
        tc_data(1800, '0.667'),
        tc_data(1800, '0.723'),
        tc_data(1460, '0.056'),
    ),
    data_row(
        tc_data(900, '0.10\u2020', 'left'),
        tc_data(1400, '0.752'),
        tc_data(2000, '0.751\u20130.753'),
        tc_data(1800, '0.661'),
        tc_data(1800, '0.727'),
        tc_data(1460, '0.066'),
    ),
    data_row(
        tc_data(900, 'FedAvg', 'left'),
        tc_data(1400, '0.757'),
        tc_data(2000, '0.756\u20130.758'),
        tc_data(1800, '0.669'),
        tc_data(1800, '0.722'),
        tc_data(1460, '0.054'),
    ),
]
table7_xml = tbl([900, 1400, 2000, 1800, 1800, 1460], *t7_rows)

# Anchor: after second Table 5 caption paragraph (L7837-7852)
# Unique anchor: the Table 5 caption note after the figure
anchor_4de = (
    '        <w:t xml:space="preserve">Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden&#x2019;s J thresholds (NHANES internal validation, n=15,650). At the global thresho'
)

tables_6_7_block = (
    p_blank() + '\n' +
    p_table_title('Table 6. Node B ablation experiment: external BRFSS performance (n=1,282,897) with and without the elderly-rural node.') + '\n' +
    table6_xml + '\n' +
    p_body('\u2020Ablation elderly gap vs full FedAvg: +0.016 (+29.1%), confirming Node B as the primary driver of fairness improvement. Centralised NN is DiabetesNet trained on all NHANES data without federation.') + '\n' +
    p_blank() + '\n' +
    p_table_title('Table 7. FedProx proximal parameter (\u03bc) sensitivity analysis on external BRFSS validation. \u2020Existing result from main analysis; all others are new experiments.') + '\n' +
    table7_xml + '\n' +
    p_body('\u03bc=0.05 achieves the best elderly gap (0.056) but lower overall AUC (0.755) than FedAvg (0.757); FedAvg remains preferred as it maximises both AUC and fairness simultaneously. Larger \u03bc values impose stronger proximal regularisation, dampening Node B\u2019s elderly-specific learning signal.') + '\n' +
    p_blank() + '\n'
)

# Insert after the closing </w:p> that contains the Table 5 caption text
# The caption paragraph is the one starting with the "Table 5. Equalized Odds..." at L7837
# After L7839 (</w:p>), there's a blank paragraph L7840-7852, then a figure
# We insert after L7852 (the blank paragraph that follows the caption)
# Anchor: end of Table 5 note text paragraph + following blank para
anchor_4de = (
    '        <w:t xml:space="preserve">Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden&#x2019;s J thresholds (NHANES internal validation, n=15,650). At the global thresho'
)

# Find this text and get its paragraph ending, then the next blank paragraph
# Replacement: after the Table 5 caption paragraph and its following blank
anchor_4de_full = (
    '        <w:t xml:space="preserve">Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden&#x2019;s J thresholds (NHANES internal validation, n=15,650). At the global thresho'
    'ld, FedProx achieves EOD=0.713 — the worst fairness — but subgroup-specific Youden&#x2019;s J thresholds reduce the EOD to 0.271, a factor of 2.6&#x00D7; lower.</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    '    <w:p>\n'
    '      <w:pPr>\n'
    '        <w:spacing w:line="480" w:lineRule="auto"/>\n'
    '      </w:pPr>\n'
    '      <w:r>\n'
    '        <w:rPr>\n'
    '          <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '          <w:sz w:val="24"/>\n'
    '          <w:szCs w:val="24"/>\n'
    '        </w:rPr>\n'
    '        <w:t xml:space="preserve"/>\n'
    '      </w:r>\n'
    '    </w:p>'
)

# check if this long anchor exists
if anchor_4de_full not in content:
    # Try the shorter version and check what follows
    print("  [4D+4E] Long anchor not found, trying pattern search...")
    # find the second occurrence of Table 5 caption
    idx = content.find('Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden')
    idx2 = content.find('Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden', idx+1)
    if idx2 > 0:
        # get context around idx2
        snip = content[idx2-50:idx2+500]
        print(f"  Found second occurrence, context: {repr(snip[:200])}")

replace_4de_full = (
    '        <w:t xml:space="preserve">Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden&#x2019;s J thresholds (NHANES internal validation, n=15,650). At the global thresho'
    'ld, FedProx achieves EOD=0.713 — the worst fairness — but subgroup-specific Youden&#x2019;s J thresholds reduce the EOD to 0.271, a factor of 2.6&#x00D7; lower.</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    '    <w:p>\n'
    '      <w:pPr>\n'
    '        <w:spacing w:line="480" w:lineRule="auto"/>\n'
    '      </w:pPr>\n'
    '      <w:r>\n'
    '        <w:rPr>\n'
    '          <w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>\n'
    '          <w:sz w:val="24"/>\n'
    '          <w:szCs w:val="24"/>\n'
    '        </w:rPr>\n'
    '        <w:t xml:space="preserve"/>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    + tables_6_7_block
)
apply(anchor_4de_full, replace_4de_full, '4D+4E (Tables 6 and 7)')


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4F: Replace RQ1–RQ4 with result-backed versions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4F] Replacing RQ1–RQ4 statements...")

apply(
    '1. (RQ1) Accuracy: Do federated models trained on demographically heterogeneous nodes match or exceed the published external AUC of 0.717?',
    '1. (RQ1) Accuracy: FedAvg achieves external AUC=0.757 (95% CI: 0.756\u20130.758) on 1,282,897 BRFSS records, exceeding both the published XGBoost baseline (0.717, +5.7 pp) and a centralised DiabetesNet trained on all NHANES data (external AUC=0.749), confirming that demographically heterogeneous FL improves generalisation beyond centralised training.',
    '4F-RQ1'
)

apply(
    '2. (RQ2) Fairness: Does federated training reduce the elderly performance gap from 0.135, specifically by leveraging Node B (elderly-rural hospital, 82.4% elderly patients)?',
    '2. (RQ2) Fairness: FedAvg reduces the external elderly fairness gap from 0.069 (centralised XGBoost) to 0.054 (\u221221.7%); Node B ablation (Nodes A+C only) widens the gap to 0.070 (+29.1%), providing causal evidence that Node B\u2019s elderly-rural demographics are the primary driver of the fairness improvement.',
    '4F-RQ2'
)

apply(
    '3. (RQ3) Aggregation Strategy: Which FL strategy\u2014FedAvg, FedProx, or FedNova\u2014performs best on non-IID clinical data?',
    '3. (RQ3) Aggregation Strategy: FedAvg (AUC=0.757) outperforms FedProx (\u03bc=0.1: AUC=0.752) and FedNova (AUC=0.744); FedProx \u03bc sensitivity analysis (\u03bc\u2208{0.01, 0.05, 0.1}) yields external AUC 0.747\u20130.755 with elderly gap 0.056\u20130.066, confirming FedAvg as the recommended strategy at the evaluated per-node data scale (\u223c3,000\u20134,500 training samples).',
    '4F-RQ3'
)

apply(
    '4. (RQ4) Privacy: How does \u03b5-differential privacy (via DP-SGD) affect the accuracy-fairness trade-off at clinical FL dataset scales?',
    '4. (RQ4) Calibration and Privacy: FedAvg\u2019s reliability analysis on BRFSS reveals systematic overconfidence (ECE=0.276, Brier score=0.217), identifying probability calibration as a deployment prerequisite; differential privacy at \u03b5\u22645 causes model collapse (AUC\u22480.5) at current per-node scales, motivating future work on larger federated cohorts (\u226540,000 per node).',
    '4F-RQ4'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4G: Replace "fundamental privacy-utility tension" (3 occurrences)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4G] Replacing 'fundamental privacy-utility tension'...")

# Occurrence 1 (L219 — in highlights): "demonstrating fundamental privacy-utility tension"
apply(
    'demonstrating fundamental privacy-utility tension',
    'demonstrating a privacy-utility trade-off observed at the evaluated per-node sample scale (\u223c3,000\u20134,500 samples)',
    '4G-1 (highlights)'
)

# Occurrence 2 (L411 — in conclusion/abstract): "The fundamental privacy-utility tension identified"
apply(
    'The fundamental privacy-utility tension identified at \u03b5 \u2264 5 motivates future work on larger federated cohorts and secure aggregation protocols.',
    'The privacy-utility trade-off observed at the evaluated per-node sample scale (\u223c3,000\u20134,500 samples) at \u03b5 \u2264 5 motivates future work on larger federated cohorts and secure aggregation protocols.',
    '4G-2 (abstract)'
)

# Occurrence 3 is in L8637 conclusion — handled below by edit 4L(3)/conclusion replacement


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4K: Section 4.4 Calibration Analysis (insert before 4.4 Fairness Analysis)
#          + renumber 4.4→4.5 (Fairness) and 4.5→4.6 (Differential Privacy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4K] Adding Section 4.4 Calibration Analysis...")

# Build calibration table (Table 8)
# Cols: Model(2200) | ECE-10bin(1400) | Brier(1200) | Reliability(1200) | Resolution(1200) | Uncertainty(1200) | n(960) = 9360
t8_rows = [
    hdr_row(
        tc_hdr(2200, 'Model / Dataset', 'left'),
        tc_hdr(1400, 'ECE (10-bin)'),
        tc_hdr(1200, 'Brier Score'),
        tc_hdr(1200, 'Reliability'),
        tc_hdr(1200, 'Resolution'),
        tc_hdr(1200, 'Uncertainty'),
        tc_hdr(960,  'n'),
    ),
    data_row(
        tc_data(2200, 'FedAvg \u2014 BRFSS external', 'left'),
        tc_data(1400, '0.276'),
        tc_data(1200, '0.217'),
        tc_data(1200, '0.113'),
        tc_data(1200, '0.012'),
        tc_data(1200, '0.115'),
        tc_data(960,  '1,282,897'),
    ),
]
table8_xml = tbl([2200, 1400, 1200, 1200, 1200, 1200, 960], *t8_rows)

calib_section = (
    p_heading_sub('4.4 Calibration Analysis') + '\n' +
    p_body(
        'The reliability of probabilistic predictions is as clinically important as discriminative accuracy. '
        'A well-calibrated model should assign predicted probabilities matching empirical event rates: a model '
        'predicting 30% diabetes risk should be correct approximately 30% of the time in patients at that risk level. '
        'We evaluated FedAvg\u2019s calibration on the full BRFSS external validation set (n=1,282,897) using Expected '
        'Calibration Error (ECE) over ten equal-width bins and Brier score decomposition (Murphy, 1973). Table 8 and '
        'Figure 12 (reliability diagram) present the results.'
    ) + '\n' +
    p_blank() + '\n' +
    p_table_title('Table 8. FedAvg calibration on BRFSS external validation (n=1,282,897). Brier decomposition: BS = Reliability \u2212 Resolution + Uncertainty.') + '\n' +
    table8_xml + '\n' +
    p_blank() + '\n' +
    p_body(
        'The ECE of 0.276 indicates substantial systematic overconfidence: FedAvg consistently assigns higher predicted '
        'probabilities than the empirical positive rates, particularly in the 0.2\u20130.9 predicted probability range '
        '(Figure 12). The Brier score decomposition reveals that the reliability component (0.113) far exceeds the '
        'resolution component (0.012), confirming that miscalibration rather than poor discrimination drives the overall '
        'Brier score of 0.217. The strong discriminative performance (AUC=0.757) coexisting with poor calibration is a '
        'common pattern in neural networks trained on class-imbalanced data without explicit calibration objectives.'
    ) + '\n' +
    p_body(
        'These findings have direct clinical implications: uncalibrated probability scores cannot be used directly for '
        'clinical decision thresholds or risk stratification. Post-hoc calibration methods \u2014 Platt scaling, '
        'isotonic regression, or temperature scaling \u2014 should be applied before clinical deployment. Future work '
        'should incorporate label smoothing or calibration losses during federated training to jointly optimise '
        'discrimination and calibration across heterogeneous nodes.'
    ) + '\n' +
    p_blank() + '\n'
)

# Insert before the old "4.4 Fairness Analysis" heading, and rename it to 4.5
anchor_4k = '<w:t xml:space="preserve">4.4 Fairness Analysis</w:t>'
replace_4k = (
    calib_section +
    p_heading_sub('4.5 Fairness Analysis').replace(
        '<w:t xml:space="preserve">4.5 Fairness Analysis</w:t>',
        '<w:t xml:space="preserve">4.5 Fairness Analysis</w:t>'
    )
).replace('PLACEHOLDER_HEADING', '<w:t xml:space="preserve">4.5 Fairness Analysis</w:t>')

# Build carefully: replace the old heading text
new_heading_45 = '<w:t xml:space="preserve">4.5 Fairness Analysis</w:t>'
apply(
    anchor_4k,
    calib_section.rstrip('\n') + '\n    ' + new_heading_45[0:len(new_heading_45)],
    '4K-heading-replace'
)

# Also rename 4.5 Differential Privacy → 4.6
apply(
    '<w:t xml:space="preserve">4.5 Differential Privacy</w:t>',
    '<w:t xml:space="preserve">4.6 Differential Privacy</w:t>',
    '4K-renumber-4.5→4.6'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4J: FedNova underperformance paragraph after Section 5.2 text
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4J] Adding FedNova underperformance paragraph after Section 5.2...")

fednova_para = p_body(
    'FedNova\u2019s underperformance (AUC=0.744, \u22121.3 pp vs FedAvg) is mechanistically explicable through its '
    '\u03c4-normalisation scheme. Although FedNova divides each client\u2019s update by its local step count before '
    'aggregation \u2014 theoretically correcting for the \u03c4 heterogeneity (\u03c4_A=5, \u03c4_B=3, \u03c4_C=4) '
    '\u2014 this normalisation implicitly down-weights Node B\u2019s contribution relative to Nodes A and C, since '
    '\u03c4_B is the smallest. Node B carries the highest concentration of elderly patients (82.4%) and the highest '
    'diabetes prevalence (28.5%), making it the primary source of the elderly-specific gradient signal that drives '
    'fairness improvement. The Node B ablation experiment (Table 6) confirms this: removing Node B from FedAvg '
    'training widened the elderly gap from 0.054 to 0.070 (+29.1%), while the overall AUC dropped from 0.757 to '
    '0.739. FedNova\u2019s \u03c4-normalisation partially replicates this ablation effect even with Node B present, '
    'explaining its intermediate fairness (gap=0.064) and lower AUC. A redesigned FedNova experiment with '
    '\u03c4_B \u2265 \u03c4_A would clarify whether the normalisation magnitude or the effective gradient weighting '
    'is the key mechanism.'
)

# Anchor: the Section 5.2 paragraph ending with "future work." then the next section heading
anchor_4j = (
    'FedProx paper&#x2019;s recommendation of \u03bc=0.01 for most settings [4] would likely recover more of Node B&#x2019;s signal while maintaining convergence \u2014 a sensitivity analysis across \u03bc \u2208 {0.01, 0.05, 0.1} is identified as a priority for future work.'
)

replace_4j = (
    'FedProx paper&#x2019;s recommendation of \u03bc=0.01 for most settings [4] would likely recover more of Node B&#x2019;s signal while maintaining convergence \u2014 a sensitivity analysis across \u03bc \u2208 {0.01, 0.05, 0.1} is identified as a priority for future work.'
    '\n' + p_blank() + '\n' + fednova_para
)
apply(anchor_4j, replace_4j, '4J (FedNova paragraph)')


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4L: Three positioning sentences
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4L] Adding positioning sentences...")

# 4L-1: After "circumvents entirely" paragraph (end of L524 paragraph)
positioning_intro = p_body(
    'The present study extends this framework to diabetes risk prediction \u2014 a condition with strong demographic '
    'variation in prevalence and clinical presentation \u2014 across three simulated institutional nodes with distinct '
    'age, urbanicity, and comorbidity distributions (Table 1). To our knowledge, this is the first federated learning '
    'comparison of FedAvg, FedProx, and FedNova on NHANES data with prospective external validation on 1.28 million '
    'BRFSS records, including a Node B ablation experiment providing causal evidence for the fairness mechanism.'
)

# The circumvents paragraph ends with "circumvents entirely." followed by </w:r></w:p>
anchor_4l1 = (
    'circumvents entirely.</w:t>\n'
    '      </w:r>\n'
    '    </w:p>'
)
replace_4l1 = (
    'circumvents entirely.</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    + positioning_intro
)
apply(anchor_4l1, replace_4l1, '4L-1 (positioning after circumvents)')

# 4L-2: After "imaging-based FL [9]" paragraph
positioning_nn = p_body(
    'The newly trained centralised DiabetesNet baseline (Section 3.3) achieved external AUC=0.749 '
    '(95% CI: 0.748\u20130.750) on BRFSS \u2014 confirming that FedAvg\u2019s advantage of 0.757 over the '
    'centralised NN\u2019s 0.749 (+0.8 pp) is attributable specifically to federated training on demographically '
    'heterogeneous nodes, not to the neural network architecture itself. This architecture-controlled comparison '
    'strengthens the causal claim that population diversity in federated training is the mechanism underlying '
    'improved external generalisation.'
)

anchor_4l2 = (
    'This generalisation benefit of population-diverse FL is consistent with prior findings in imaging-based FL [9].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>'
)
replace_4l2 = (
    'This generalisation benefit of population-diverse FL is consistent with prior findings in imaging-based FL [9].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    + positioning_nn
)
apply(anchor_4l2, replace_4l2, '4L-2 (positioning after imaging-based FL)')

# 4L-3: Replace conclusion paragraph (also handles 4G-3)
new_conclusion = (
    'We demonstrated that federated learning across three demographically heterogeneous simulated hospital nodes '
    'yields diabetes prediction models that substantially exceed both the published external AUC (0.717) and a '
    'centralised DiabetesNet baseline (0.749) trained on all NHANES data, with FedAvg achieving 0.757 '
    '(95% CI: 0.756\u20130.758) on 1.28 million BRFSS records. The causal mechanism for FedAvg\u2019s fairness '
    'advantage was confirmed by Node B ablation: removing the elderly-rural node widened the elderly fairness gap '
    'from 0.054 to 0.070 (+29.1%). FedProx \u03bc sensitivity analysis across \u03bc\u2208{0.01, 0.05, 0.1} yielded '
    'external AUC 0.747\u20130.755, confirming FedAvg as the preferred strategy at this dataset scale. Beyond '
    'absolute performance, FedAvg exhibits 2.2\u00d7 better distributional robustness '
    '(\u0394Int\u2192Ext=0.031 vs 0.069 for centralised) without sharing any raw patient data. Calibration '
    'analysis on BRFSS reveals substantial overconfidence (ECE=0.276, Brier=0.217), identifying probability '
    'calibration as a critical prerequisite alongside the privacy-utility trade-off observed at the evaluated '
    'per-node sample scale (\u223c3,000\u20134,500 samples) for clinical deployment readiness.'
)

old_conclusion = (
    'We demonstrated that federated learning across three demographically heterogeneous simulated hospital nodes yields diabetes prediction models that substantially exceed the published external AUC (0.717) of a centralised XGBoost baseline, with FedAvg achieving 0.757 (95% CI: 0.756\u20130.758) on 1.28 million BRFSS records. Beyond absolute performance, FedAvg exhibits 2.2\u00d7 better distributional robustness (\u0394Int\u2192Ext=0.031 vs 0.069 for centralised) and reduces the external elderly fairness gap by 21.7% (0.069\u21920.054) without sharing any raw patient data. The finding that global-threshold EOD overestimates unfairness by a factor of 2.6\u00d7 compared to subgroup-specific thresholds (0.713\u21920.271) provides a methodological caution for fairness evaluation in class-imbalanced clinical prediction settings. The fundamental privacy-utility tension \u2014 model collapse at \u03b5 \u2264 5 with current dataset sizes \u2014 motivates future work combining larger federated cohorts (\u226540,000 per node), differential privacy accounting improvements, and node-specific threshold calibration as prerequisites for clinical deployment.'
)
apply(old_conclusion, new_conclusion, '4L-3 (new conclusion)')


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4I: Replace Limitations first paragraph with fuller version
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4I] Replacing Limitations paragraph...")

old_limitations = (
    'First, the three hospital nodes are simulated from NHANES data; real-world FL faces additional heterogeneity from EHR coding practices and label definitions. Second, the NHANES-BRFSS feature mapping requires encoding assumptions for smoking and physical activity that introduce measurement invariance risk. Third, only eight features were used, excluding potentially important biomarkers (HbA1c, fasting glucose) that may not be universally available across federated nodes. Fourth, FL simulations were conducted in a single-process environment rather than true distributed infrastructure, excluding communication overhead from runtime estimates.'
)

new_limitations = (
    'The principal limitation of this study is that all three institutional nodes are simulated by partitioning a '
    'single national survey (NHANES 2015\u20132020) by demographic strata rather than representing genuinely '
    'independent clinical sites with distinct EHR systems, coding practices, and label definitions. Real-world '
    'federated networks face additional heterogeneity \u2014 including differences in ICD coding, laboratory '
    'reference ranges, and clinical documentation practices \u2014 that cannot be captured by demographic '
    'partitioning alone. The simulated design means that the privacy guarantee is definitional (raw data never '
    'leaves nodes) but not cryptographically enforced; secure aggregation protocols (e.g., SecAgg) would be '
    'required for clinical deployment. Second, calibration analysis reveals substantial overconfidence '
    '(ECE=0.276; Section 4.4), indicating that probability outputs should not be used directly for clinical '
    'risk stratification without post-hoc calibration such as Platt scaling or isotonic regression. Third, '
    'the NHANES-BRFSS feature mapping requires encoding assumptions for smoking status and physical activity '
    'that introduce measurement invariance risk; alternative encodings may alter calibration and subgroup '
    'performance. Fourth, only eight routinely collectable features were used, excluding biomarkers '
    '(HbA1c, fasting glucose) that carry substantial predictive power but may not be universally available '
    'across federated nodes. Fifth, FL simulations were conducted in a single-process environment rather than '
    'true distributed infrastructure, excluding communication overhead, network latency, and stragglers from '
    'runtime estimates. Collectively, these limitations mean that the results establish proof-of-concept for '
    'the federated learning framework rather than direct evidence of clinical deployability.'
)
apply(old_limitations, new_limitations, '4I (Limitations)')


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4M: FL+Fairness paragraph at end of Section 2.2
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4M] Adding FL+Fairness paragraph at end of Section 2.2...")

fl_fairness_para = p_body(
    'Federated learning can actively improve algorithmic fairness by incorporating demographically diverse '
    'training nodes into the aggregation process. When node distributions are systematically aligned with '
    'underrepresented subpopulations \u2014 for example, by designating a node to serve a predominantly '
    'elderly rural population \u2014 the global model receives gradient updates calibrated to that group\u2019s '
    'risk profile during every communication round. This stands in contrast to centralised training, where '
    'demographic class imbalance may cause the optimisation to allocate disproportionate capacity to '
    'majority-group patterns. The present study directly tests this mechanism by constructing Node B to '
    'serve 82.4% elderly patients (age \u226560) and subsequently ablating it from the federated pool '
    '(Table 6), providing the first experimental evidence that a demographically specialised node is '
    'causally responsible for elderly fairness improvement in a federated diabetes prediction framework.'
)

# Anchor: the last paragraph of Section 2.2 (Obermeyer et al. paragraph) ending at L711
anchor_4m = (
    'Obermeyer et al. (2019) demonstrated systematic racial bias in commercial healthcare algorithms [10]. For diabetes specifically, Gianfrancesco et al. (2018) documented performance disparities across race, age, and socioeconomic status [18]. The TRIPOD-AI reporting guidelines require explicit fairness evaluation across demographic subgroups [11].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>'
)
replace_4m = (
    'Obermeyer et al. (2019) demonstrated systematic racial bias in commercial healthcare algorithms [10]. For diabetes specifically, Gianfrancesco et al. (2018) documented performance disparities across race, age, and socioeconomic status [18]. The TRIPOD-AI reporting guidelines require explicit fairness evaluation across demographic subgroups [11].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    + fl_fairness_para
)
apply(anchor_4m, replace_4m, '4M (FL+Fairness paragraph)')


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Edits applied: {len(edits_applied)}")
for e in edits_applied:
    print(f"    \u2713 {e}")
print(f"  Content length: {original_len:,} \u2192 {len(content):,} (+{len(content)-original_len:,} chars)")

# Write output
with open(DOC, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"\n  Saved: {DOC}")
print(f"{'='*65}")
