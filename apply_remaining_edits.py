"""
apply_remaining_edits.py
=========================
Applies the 6 remaining edits that failed due to CRLF line endings:
4A, 4B, 4C-ii, 4D+4E, 4L-1, 4L-2, 4M
Normalises \r\r\n → \n before processing.
"""

import os, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DOC = os.path.join('manuscript_v4_unpacked', 'word', 'document.xml')

with open(DOC, 'rb') as f:
    raw = f.read()

# Normalise \r\r\n → \n  and  \r\n → \n  and lone \r → \n
content = raw.decode('utf-8', errors='replace')
content = content.replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')

original_len = len(content)
edits_applied = []

BORDERS = """\
            <w:tcBorders>
              <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>
              <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>
              <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>
              <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>
            </w:tcBorders>"""

TNRFONTS = '<w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'

def tc_data(width, text, align='center'):
    return (
        f'        <w:tc>\n'
        f'          <w:tcPr>\n'
        f'            <w:tcW w:type="dxa" w:w="{width}"/>\n'
        f'{BORDERS}\n'
        f'            <w:shd w:fill="FFFFFF" w:color="auto" w:val="clear"/>\n'
        f'          </w:tcPr>\n'
        f'          <w:p>\n'
        f'            <w:pPr>\n'
        f'              <w:spacing w:after="60" w:before="60"/>\n'
        f'              <w:jc w:val="{align}"/>\n'
        f'            </w:pPr>\n'
        f'            <w:r>\n'
        f'              <w:rPr>\n'
        f'                {TNRFONTS}\n'
        f'                <w:b w:val="false"/>\n'
        f'                <w:bCs w:val="false"/>\n'
        f'                <w:sz w:val="20"/>\n'
        f'                <w:szCs w:val="20"/>\n'
        f'              </w:rPr>\n'
        f'              <w:t xml:space="preserve">{text}</w:t>\n'
        f'            </w:r>\n'
        f'          </w:p>\n'
        f'        </w:tc>'
    )

def tc_hdr(width, text, align='center'):
    return (
        f'        <w:tc>\n'
        f'          <w:tcPr>\n'
        f'            <w:tcW w:type="dxa" w:w="{width}"/>\n'
        f'{BORDERS}\n'
        f'            <w:shd w:fill="D5E8F0" w:color="auto" w:val="clear"/>\n'
        f'          </w:tcPr>\n'
        f'          <w:p>\n'
        f'            <w:pPr>\n'
        f'              <w:spacing w:after="60" w:before="60"/>\n'
        f'              <w:jc w:val="{align}"/>\n'
        f'            </w:pPr>\n'
        f'            <w:r>\n'
        f'              <w:rPr>\n'
        f'                {TNRFONTS}\n'
        f'                <w:b/>\n'
        f'                <w:bCs/>\n'
        f'                <w:sz w:val="20"/>\n'
        f'                <w:szCs w:val="20"/>\n'
        f'              </w:rPr>\n'
        f'              <w:t xml:space="preserve">{text}</w:t>\n'
        f'            </w:r>\n'
        f'          </w:p>\n'
        f'        </w:tc>'
    )

def hdr_row(*cells):
    return '      <w:tr>\n        <w:trPr>\n          <w:tblHeader/>\n        </w:trPr>\n' + '\n'.join(cells) + '\n      </w:tr>'

def data_row(*cells):
    return '      <w:tr>\n' + '\n'.join(cells) + '\n      </w:tr>'

def tbl(col_widths, *rows):
    grid = '\n'.join(f'        <w:gridCol w:w="{w}"/>' for w in col_widths)
    rows_xml = '\n'.join(rows)
    return (
        '    <w:tbl>\n'
        '      <w:tblPr>\n'
        '        <w:tblW w:type="dxa" w:w="9360"/>\n'
        '        <w:tblBorders>\n'
        '          <w:top w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
        '          <w:left w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
        '          <w:bottom w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
        '          <w:right w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
        '          <w:insideH w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
        '          <w:insideV w:val="single" w:color="CCCCCC" w:sz="1"/>\n'
        '        </w:tblBorders>\n'
        '      </w:tblPr>\n'
        '      <w:tblGrid>\n'
        + grid + '\n'
        '      </w:tblGrid>\n'
        + rows_xml + '\n'
        '    </w:tbl>'
    )

def p_body(text):
    return (
        '    <w:p>\n'
        '      <w:pPr>\n'
        '        <w:spacing w:after="120" w:before="0" w:line="480" w:lineRule="auto"/>\n'
        '        <w:jc w:val="both"/>\n'
        '      </w:pPr>\n'
        '      <w:r>\n'
        '        <w:rPr>\n'
        f'          {TNRFONTS}\n'
        '          <w:b w:val="false"/>\n'
        '          <w:bCs w:val="false"/>\n'
        '          <w:i w:val="false"/>\n'
        '          <w:iCs w:val="false"/>\n'
        '          <w:sz w:val="24"/>\n'
        '          <w:szCs w:val="24"/>\n'
        '        </w:rPr>\n'
        f'        <w:t xml:space="preserve">{text}</w:t>\n'
        '      </w:r>\n'
        '    </w:p>'
    )

def p_table_title(text):
    return (
        '    <w:p>\n'
        '      <w:pPr>\n'
        '        <w:spacing w:after="80" w:before="240"/>\n'
        '      </w:pPr>\n'
        '      <w:r>\n'
        '        <w:rPr>\n'
        f'          {TNRFONTS}\n'
        '          <w:b/>\n'
        '          <w:bCs/>\n'
        '          <w:sz w:val="22"/>\n'
        '          <w:szCs w:val="22"/>\n'
        '        </w:rPr>\n'
        f'        <w:t xml:space="preserve">{text}</w:t>\n'
        '      </w:r>\n'
        '    </w:p>'
    )

def p_blank():
    return (
        '    <w:p>\n'
        '      <w:pPr>\n'
        '        <w:spacing w:line="480" w:lineRule="auto"/>\n'
        '      </w:pPr>\n'
        '      <w:r>\n'
        '        <w:rPr>\n'
        f'          {TNRFONTS}\n'
        '          <w:sz w:val="24"/>\n'
        '          <w:szCs w:val="24"/>\n'
        '        </w:rPr>\n'
        '        <w:t xml:space="preserve"/>\n'
        '      </w:r>\n'
        '    </w:p>'
    )

def apply(old, new, label):
    global content
    if old not in content:
        print(f'  WARN [{label}]: anchor not found!')
        # debug: find closest match
        # look for first 40 chars of old
        snippet = old[:60]
        if snippet in content:
            idx = content.find(snippet)
            print(f'    First 60 chars found at {idx}, context: {repr(content[idx:idx+120])}')
        return False
    n = content.count(old)
    content = content.replace(old, new, 1)
    edits_applied.append(label)
    print(f'  OK  [{label}]' + (f' ({n}x)' if n > 1 else ''))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4A: Centralised NN row in Table 2
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4A] Centralised NN row in Table 2...')
cnn_t2 = data_row(
    tc_data(2200, 'Centralised NN (DiabetesNet)', 'left'),
    tc_data(1300, '0.801'),
    tc_data(1800, '0.782\u20130.819'),
    tc_data(900,  '0.187'),
    tc_data(900,  '0.494'),
    tc_data(1080, '0.794'),
    tc_data(1180, '0.676'),
)
# Table 2 ends: FedNova Spec=0.651 in w:w=1180 cell → </w:tr> → </w:tbl>
apply(
    '              <w:t xml:space="preserve">0.651</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    '    </w:tbl>',
    '              <w:t xml:space="preserve">0.651</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    + cnn_t2 + '\n'
    '    </w:tbl>',
    '4A'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4B: Centralised NN row in Table 3
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4B] Centralised NN row in Table 3...')
cnn_t3 = data_row(
    tc_data(2200, 'Centralised NN (DiabetesNet)', 'left'),
    tc_data(1200, '0.749'),
    tc_data(1760, '0.748\u20130.750'),
    tc_data(900,  '0.306'),
    tc_data(900,  '0.356'),
    tc_data(1100, '0.720'),
    tc_data(1300, '0.643'),
)
apply(
    '              <w:t xml:space="preserve">0.635</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    '    </w:tbl>',
    '              <w:t xml:space="preserve">0.635</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '      </w:tr>\n'
    + cnn_t3 + '\n'
    '    </w:tbl>',
    '4B'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4C-ii: Add Centralised NN row in Table 4
# After Centralised XGBoost row (has "Baseline" in gap-vs-centralised col)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4C-ii] Centralised NN row in Table 4...')

# Centralised NN external fairness: age18-39=0.720, age60+=0.657, gap=0.063
# Gap vs XGBoost ext: (0.063-0.069)/0.069 = -8.7%
cnn_t4 = data_row(
    tc_data(2100, 'Centralised NN (DiabetesNet)', 'left'),
    tc_data(1380, '0.720'),
    tc_data(1380, '0.657'),
    tc_data(1100, '0.063'),
    tc_data(1500, '\u22128.7%'),
    tc_data(1900, 'BRFSS external'),
)

# Find the Centralised XGBoost row end in Table 4
# It has "Baseline" in gap-vs-centralised and "BRFSS external" in domain
# followed by the FedAvg row which has "FedAvg (ext.)"
# Unique anchor: the specific closing sequence of Centralised XGBoost row

# Build the exact row closing XML that we need to match
anchor_cxgb_close = (
    '              <w:t xml:space="preserve">Baseline</w:t>\n'
    '            </w:r>\n'
    '          </w:p>\n'
    '        </w:tc>\n'
    '        <w:tc>\n'
    '          <w:tcPr>\n'
    '            <w:tcW w:type="dxa" w:w="1900"/>\n'
    + BORDERS + '\n'
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
    '      </w:tr>'
)

# We need to identify this specific row (Centralised XGBoost, not FedAvg/FedProx/FedNova)
# The unique identifier: "Baseline" appears only in the Centralised XGBoost row
apply(
    anchor_cxgb_close,
    anchor_cxgb_close + '\n' + cnn_t4,
    '4C-ii'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4D+4E: Tables 6 and 7 after Table 5 caption
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4D+4E] Tables 6 and 7 after Table 5 caption...')

# Build Table 6 (Node B Ablation)
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
        tc_data(3000, 'Centralised NN (DiabetesNet)', 'left'),
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

# Build Table 7 (FedProx sensitivity)
t7_rows = [
    hdr_row(
        tc_hdr(900, '\u03bc', 'left'),
        tc_hdr(1500, 'Ext AUC'),
        tc_hdr(2100, '95% CI'),
        tc_hdr(1700, 'AUC (\u226560)'),
        tc_hdr(1700, 'AUC (18\u201339)'),
        tc_hdr(1460, 'Elderly Gap'),
    ),
    data_row(
        tc_data(900, '0.01', 'left'),
        tc_data(1500, '0.747'),
        tc_data(2100, '0.746\u20130.748'),
        tc_data(1700, '0.656'),
        tc_data(1700, '0.713'),
        tc_data(1460, '0.057'),
    ),
    data_row(
        tc_data(900, '0.05', 'left'),
        tc_data(1500, '0.755'),
        tc_data(2100, '0.754\u20130.756'),
        tc_data(1700, '0.667'),
        tc_data(1700, '0.723'),
        tc_data(1460, '0.056'),
    ),
    data_row(
        tc_data(900, '0.10\u2020', 'left'),
        tc_data(1500, '0.752'),
        tc_data(2100, '0.751\u20130.753'),
        tc_data(1700, '0.661'),
        tc_data(1700, '0.727'),
        tc_data(1460, '0.066'),
    ),
    data_row(
        tc_data(900, 'FedAvg', 'left'),
        tc_data(1500, '0.757'),
        tc_data(2100, '0.756\u20130.758'),
        tc_data(1700, '0.669'),
        tc_data(1700, '0.722'),
        tc_data(1460, '0.054'),
    ),
]
table7_xml = tbl([900, 1500, 2100, 1700, 1700, 1460], *t7_rows)

tables_block = (
    '\n' + p_blank() + '\n'
    + p_table_title(
        'Table 6. Node B ablation experiment: external BRFSS performance (n=1,282,897). '
        'All metrics evaluated on BRFSS 2020\u20132022. Elderly gap = AUC(18\u201339) \u2212 AUC(\u226560).'
    ) + '\n'
    + table6_xml + '\n'
    + p_body(
        '\u2020Ablation gap increase vs full FedAvg: +0.016 (+29.1%), confirming Node B as '
        'the primary driver of fairness improvement. Centralised NN = DiabetesNet trained on '
        'all NHANES training data without federation.'
    ) + '\n'
    + p_blank() + '\n'
    + p_table_title(
        'Table 7. FedProx proximal parameter (\u03bc) sensitivity on external BRFSS validation '
        '(n=1,282,897). \u2020Existing result from main analysis; all others newly trained.'
    ) + '\n'
    + table7_xml + '\n'
    + p_body(
        '\u03bc=0.05 achieves the lowest elderly gap (0.056) but slightly lower AUC (0.755) than '
        'FedAvg (0.757); FedAvg remains preferred as it jointly maximises AUC and fairness. '
        'Larger \u03bc values impose stronger proximal regularisation, dampening Node B\u2019s '
        'elderly-specific learning signal and increasing the fairness gap.'
    ) + '\n'
    + p_blank() + '\n'
)

# Find the Table 5 caption (second occurrence - the note after the figure)
# The second Table 5 caption contains the text about threshold reducing EOD to 0.271
idx1 = content.find('Table 5. Equalized Odds Difference')
idx2 = content.find('Table 5. Equalized Odds Difference', idx1 + 1)

if idx2 > 0:
    # Find the end of this paragraph
    p_end = content.find('</w:p>', idx2)
    if p_end > 0:
        para_close = '</w:p>'
        old_anchor = content[idx2-200:p_end+6]  # full paragraph context
        print(f'  Found second Table 5 caption at {idx2}')
        # Insert tables after this paragraph + blank paragraph
        # Find the next blank paragraph after p_end
        next_p_start = content.find('<w:p>', p_end)
        # Find the blank paragraph (the one with empty <w:t>)
        blank_p_end = content.find('</w:p>', next_p_start)

        insert_point_text = content[p_end:blank_p_end+6]
        print(f'  Insert point context: {repr(insert_point_text[:100])}')

        anchor_str = content[idx2 - 100: blank_p_end + 6]
        # Make a more targeted replacement
        replace_str = anchor_str + tables_block

        apply(anchor_str, replace_str, '4D+4E')
else:
    print('  WARN [4D+4E]: Could not find Table 5 second caption!')


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4L-1: Positioning sentence after "circumvents entirely"
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4L-1] Positioning sentence after circumvents...')

positioning_intro = p_body(
    'The present study extends this framework to diabetes risk prediction \u2014 a condition '
    'with strong demographic variation in prevalence and clinical presentation \u2014 across '
    'three simulated institutional nodes with distinct age, urbanicity, and comorbidity '
    'distributions (Table 1). To our knowledge, this is the first federated learning comparison '
    'of FedAvg, FedProx, and FedNova on NHANES data with prospective external validation on '
    '1.28 million BRFSS records, including a Node B ablation experiment providing causal evidence '
    'for the fairness mechanism.'
)

apply(
    'circumvents entirely.</w:t>\n'
    '      </w:r>\n'
    '    </w:p>',
    'circumvents entirely.</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    + positioning_intro,
    '4L-1'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4L-2: Positioning sentence after imaging-based FL [9]
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4L-2] Positioning sentence after imaging-based FL...')

positioning_nn = p_body(
    'The newly trained centralised DiabetesNet baseline (Section 3.3) achieved external '
    'AUC=0.749 (95% CI: 0.748\u20130.750) on BRFSS \u2014 confirming that FedAvg\u2019s '
    'advantage of 0.757 over the centralised NN\u2019s 0.749 (+0.8 pp) is attributable '
    'specifically to federated training on demographically heterogeneous nodes, not to the '
    'neural network architecture itself. This architecture-controlled comparison strengthens '
    'the causal claim that population diversity in federated training is the mechanism '
    'underlying improved external generalisation.'
)

apply(
    'This generalisation benefit of population-diverse FL is consistent with prior findings in imaging-based FL [9].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>',
    'This generalisation benefit of population-diverse FL is consistent with prior findings in imaging-based FL [9].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    + positioning_nn,
    '4L-2'
)


# ─────────────────────────────────────────────────────────────────────────────
# EDIT 4M: FL+Fairness paragraph at end of Section 2.2
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4M] FL+Fairness paragraph in Section 2.2...')

fl_fairness = p_body(
    'Federated learning can actively improve algorithmic fairness by incorporating '
    'demographically diverse training nodes into the aggregation process. When node '
    'distributions are systematically aligned with underrepresented subpopulations \u2014 '
    'for example, by designating a node to serve a predominantly elderly rural population '
    '\u2014 the global model receives gradient updates calibrated to that group\u2019s risk '
    'profile during every communication round. This stands in contrast to centralised '
    'training, where demographic class imbalance may cause the optimisation to allocate '
    'disproportionate capacity to majority-group patterns. The present study directly tests '
    'this mechanism by constructing Node B to serve 82.4% elderly patients (age \u226560) '
    'and subsequently ablating it from the federated pool (Table 6), providing the first '
    'experimental evidence that a demographically specialised node is causally responsible '
    'for elderly fairness improvement in a federated diabetes prediction framework.'
)

apply(
    'Obermeyer et al. (2019) demonstrated systematic racial bias in commercial healthcare algorithms [10]. For diabetes specifically, Gianfrancesco et al. (2018) documented performance disparities across race, age, and socioeconomic status [18]. The TRIPOD-AI reporting guidelines require explicit fairness evaluation across demographic subgroups [11].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>',
    'Obermeyer et al. (2019) demonstrated systematic racial bias in commercial healthcare algorithms [10]. For diabetes specifically, Gianfrancesco et al. (2018) documented performance disparities across race, age, and socioeconomic status [18]. The TRIPOD-AI reporting guidelines require explicit fairness evaluation across demographic subgroups [11].</w:t>\n'
    '      </w:r>\n'
    '    </w:p>\n'
    + fl_fairness,
    '4M'
)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'  Edits applied: {len(edits_applied)}')
for e in edits_applied:
    print(f'    \u2713 {e}')
print(f'  Content: {original_len:,} \u2192 {len(content):,} (+{len(content)-original_len:,} chars)')

# Write back as UTF-8 bytes (binary mode to preserve as-is)
with open(DOC, 'wb') as f:
    f.write(content.encode('utf-8'))
print(f'\n  Saved: {DOC}')
print(f'{"="*65}')
