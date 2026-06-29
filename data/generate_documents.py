"""Generates synthetic MSAs / vendor agreements for the retrieval benchmark.

Real client contracts can't be used here for obvious reasons, so each doc is
built from a pool of boilerplate sections (recitals, definitions, payment
terms, confidentiality, etc.) plus a handful of "needle" clauses that vary
per document -- indemnification scope, IP ownership, liability caps, term
length. Those needle clauses are what the benchmark queries actually target.

Each section is written at realistic single-clause length (roughly 300-500
words, matching how dense real MSA clauses actually read) and chunking never
merges two sections together. One clause = one chunk, always. Earlier drafts
greedily packed multiple unrelated sections into a single chunk to hit a
word-count target, which produced bigger chunks but defeated the entire
point of the project -- it just buried the needle clause inside a pile of
unrelated boilerplate instead of keeping it isolated.
"""

import json
import random
from pathlib import Path

random.seed(7)

COMPANIES = [
    "Brightfield Logistics Inc.", "Aurellia Pharmaceuticals Corp.", "Northgate Materials LLC",
    "Vantia Cloud Systems Inc.", "Redline Industrial Group", "Cascade Biotech Partners LLC",
    "Solace Energy Holdings Inc.", "Marlowe & Pierce Consulting LLC", "Tundra Robotics Corp.",
    "Ferrovia Rail Services Inc.", "Glasswing Media Group LLC", "Korrigan Defense Systems Inc.",
]

BOILERPLATE_SECTIONS = [
    """1. RECITALS

WHEREAS, the parties wish to set forth the terms under which Service Provider will furnish
certain services to Client as described in one or more Statements of Work executed under this
Master Services Agreement (the "Agreement"); and WHEREAS, the parties acknowledge that any
Statement of Work executed hereunder shall be governed by the terms of this Agreement unless
expressly stated otherwise therein, and that in the event of any conflict between the terms of
this Agreement and the terms of a Statement of Work, the terms of this Agreement shall control
unless the Statement of Work expressly and specifically states that it is amending a named
provision of this Agreement; and WHEREAS, Client desires to engage Service Provider, and
Service Provider desires to be engaged by Client, on the terms and subject to the conditions
set forth herein, including the scope of services, the compensation payable therefor, and the
allocation of risk between the parties as more fully described in the sections that follow;
NOW, THEREFORE, in consideration of the mutual covenants contained herein, and for other good
and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the
parties agree as follows.""",
    """2. DEFINITIONS

For purposes of this Agreement, the following terms shall have the meanings set forth below.
"Affiliate" means, with respect to any entity, any other entity that directly or indirectly
controls, is controlled by, or is under common control with such entity, where "control" means
the possession, directly or indirectly, of the power to direct or cause the direction of the
management and policies of such entity, whether through ownership of voting securities, by
contract, or otherwise. "Confidential Information" means any non-public information disclosed
by either party, whether orally, visually, or in written or electronic form, that is designated
as confidential at the time of disclosure or that reasonably should be understood to be
confidential given the nature of the information and the circumstances of disclosure, including
without limitation pricing, business plans, technical specifications, and the terms of this
Agreement itself. "Deliverables" means the work product, software, documentation, reports, or
other materials to be provided by Service Provider as specified in an applicable Statement of
Work, including any modifications, enhancements, or derivative works thereof created during the
term of this Agreement. "Statement of Work" or "SOW" means a written document executed by both
parties that describes the specific services, Deliverables, fees, and timeline for a particular
engagement under this Agreement.""",
    """4. PAYMENT TERMS

Client shall pay all undisputed invoiced amounts within thirty (30) days of the invoice date.
Any amount not paid when due shall accrue interest at the lesser of one and one-half percent
(1.5%) per month or the maximum rate permitted by applicable law, calculated from the original
due date until the date payment is actually received. Service Provider may suspend performance
of services upon thirty (30) days' written notice of non-payment, provided Client has not cured
such non-payment within the notice period, and any such suspension shall not relieve Client of
its obligation to pay for services already rendered or expenses already incurred prior to the
suspension. All fees are exclusive of taxes, duties, and similar governmental assessments,
which shall be borne by Client unless Client furnishes a valid exemption certificate prior to
invoicing. Client may dispute any portion of an invoice in good faith by providing written
notice within fifteen (15) days of receipt, specifying the disputed amount and the basis for
the dispute, and the parties shall use reasonable efforts to resolve any such dispute within
thirty (30) days; undisputed amounts shall remain due and payable in accordance with the
schedule set forth above notwithstanding any ongoing dispute as to other amounts.""",
    """5. CONFIDENTIALITY

Each party agrees to hold the other party's Confidential Information in strict confidence and
not to disclose such Confidential Information to any third party without the prior written
consent of the disclosing party, except to employees, contractors, and advisors who have a need
to know such information in connection with the performance of this Agreement and who are bound
by confidentiality obligations no less restrictive than those contained herein. Each party shall
use the same degree of care to protect the other party's Confidential Information as it uses to
protect its own confidential information of similar nature, but in no event less than a
reasonable degree of care. The obligations of confidentiality shall not apply to information
that is or becomes publicly available through no fault of the receiving party, was already
known to the receiving party prior to disclosure, is independently developed by the receiving
party without reference to the disclosing party's Confidential Information, or is required to
be disclosed by law or court order, provided the receiving party gives prompt notice to allow
the disclosing party to seek a protective order. The obligations of confidentiality shall
survive termination of this Agreement for a period of five (5) years, except with respect to
trade secrets, which shall remain protected for as long as they retain trade secret status under
applicable law.""",
    """7. WARRANTIES

Service Provider warrants that the services will be performed in a professional and workmanlike
manner consistent with generally accepted industry standards by personnel possessing the skill
and experience customarily exercised by similarly situated service providers. Service Provider
further warrants that the Deliverables, as delivered, will materially conform to the
specifications set forth in the applicable Statement of Work for a period of ninety (90) days
following delivery. EXCEPT AS EXPRESSLY SET FORTH IN THIS SECTION, SERVICE PROVIDER MAKES NO
OTHER WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT, AND ALL SUCH WARRANTIES
ARE HEREBY EXPRESSLY DISCLAIMED. Client's sole and exclusive remedy for breach of the foregoing
warranty shall be re-performance of the non-conforming services or re-delivery of the
non-conforming Deliverables at no additional charge, provided Client notifies Service Provider
of the non-conformance in writing within the warranty period described above.""",
    """9. TERMINATION

Either party may terminate this Agreement for convenience upon ninety (90) days' prior written
notice to the other party, in which case any Statements of Work then in progress shall continue
to be governed by this Agreement through their natural completion unless the parties otherwise
agree in writing. Either party may terminate this Agreement immediately upon written notice if
the other party materially breaches this Agreement and fails to cure such breach within thirty
(30) days of receiving written notice describing the breach in reasonable detail, or
immediately upon written notice if the other party becomes insolvent, files for bankruptcy
protection, or has a receiver appointed over a substantial portion of its assets. Upon
termination for any reason, Client shall pay Service Provider for all services performed and
expenses incurred through the effective date of termination, and each party shall promptly
return or destroy, at the disclosing party's election, all Confidential Information of the
other party in its possession.""",
    """11. GOVERNING LAW AND DISPUTE RESOLUTION

This Agreement shall be governed by and construed in accordance with the laws of the State of
Delaware, without regard to its conflict of laws principles that would result in the application
of the laws of any other jurisdiction. Any dispute arising out of or relating to this Agreement
shall first be submitted to good-faith negotiation between senior executives of each party with
authority to resolve the dispute, and if unresolved within thirty (30) days of the initial
written notice of the dispute, shall be settled by binding arbitration administered by the
American Arbitration Association in Wilmington, Delaware, in accordance with its Commercial
Arbitration Rules then in effect. The arbitration shall be conducted by a single arbitrator,
and judgment on the award rendered may be entered in any court having jurisdiction thereof.
Notwithstanding the foregoing, either party may seek injunctive or other equitable relief in a
court of competent jurisdiction at any time to prevent actual or threatened infringement,
misappropriation, or violation of a party's intellectual property rights or Confidential
Information.""",
    """12. MISCELLANEOUS

This Agreement, together with all Statements of Work executed hereunder, constitutes the entire
agreement between the parties and supersedes all prior agreements and understandings, whether
written or oral, relating to the subject matter herein. No waiver of any provision of this
Agreement shall be effective unless in writing and signed by the waiving party, and no failure
or delay by either party in exercising any right hereunder shall operate as a waiver thereof.
Neither party may assign this Agreement without the prior written consent of the other party,
except that either party may assign this Agreement without consent in connection with a merger,
acquisition, or sale of substantially all of its assets. If any provision of this Agreement is
held invalid or unenforceable by a court of competent jurisdiction, the remainder of the
Agreement shall continue in full force and effect, and the parties shall negotiate in good
faith to replace the invalid provision with a valid provision that most closely approximates
the original intent of the parties.""",
]

# Each needle is a (clause_title, clause_text) tuple. The benchmark queries
# in src/benchmark.py are written against this exact pool, so don't reorder
# without updating those queries too.
INDEMNIFICATION_NEEDLES = [
    """6. INDEMNIFICATION

Service Provider shall defend, indemnify, and hold harmless Client and its officers, directors,
and employees from and against any third-party claim, suit, or proceeding alleging that the
Deliverables, as provided by Service Provider and used in accordance with this Agreement,
infringe a valid patent, copyright, or trade secret of a third party, provided that Client
promptly notifies Service Provider in writing of such claim, grants Service Provider sole
control of the defense and settlement thereof, and provides reasonable cooperation at Service
Provider's expense. Service Provider's indemnification obligation under this Section shall not
apply to claims arising from Client's modification of the Deliverables, Client's combination of
the Deliverables with materials not supplied by Service Provider, or Client's use of the
Deliverables in a manner not contemplated by the applicable Statement of Work, in each case
where the infringement would not have occurred but for such modification, combination, or use.
Should the Deliverables become, or in Service Provider's reasonable judgment be likely to
become, the subject of an infringement claim, Service Provider may, at its option and expense,
either procure for Client the right to continue using the Deliverables, replace or modify the
Deliverables to be non-infringing while providing substantially equivalent functionality, or
refund the fees paid by Client for the affected Deliverables.""",
    """6. INDEMNIFICATION

Each party (the "Indemnifying Party") shall indemnify, defend, and hold harmless the other party
from third-party claims arising out of the Indemnifying Party's gross negligence or willful
misconduct in the performance of this Agreement. Notwithstanding the foregoing, Service
Provider's indemnification obligations with respect to third-party intellectual property
infringement claims shall be capped at an amount equal to the fees paid by Client under the
applicable Statement of Work during the twelve (12) months preceding the claim, and shall
exclude any claim arising from open-source components disclosed to Client prior to delivery or
from third-party materials that Client specifically directed Service Provider to incorporate
into the Deliverables. The indemnifying party's obligations under this Section are conditioned
on the indemnified party providing prompt written notice of the claim, reasonable cooperation in
the defense at the indemnifying party's expense, and sole control over the defense and any
settlement, provided that no settlement imposing liability or obligations on the indemnified
party shall be entered into without its prior written consent, such consent not to be
unreasonably withheld or delayed.""",
    """6. INDEMNIFICATION

Service Provider shall indemnify Client against losses, damages, and reasonable attorneys' fees
arising from a third party's claim that the Deliverables misappropriate such third party's trade
secrets or infringe such third party's patent rights, but only to the extent such infringement
arises from Service Provider's own development work and not from designs, specifications, or
components furnished by Client or from Client's combination of the Deliverables with other
products or services not provided by Service Provider. This indemnification obligation is the
parties' sole and exclusive remedy for intellectual property infringement claims and is granted
in lieu of all other remedies, whether in contract, tort, or otherwise, and is subject to a
cap equal to two (2) times the aggregate fees paid by Client under the applicable Statement of
Work, except where the infringement resulted from Service Provider's willful misconduct, in
which case no such cap shall apply.""",
]

LIABILITY_NEEDLES = [
    """8. LIMITATION OF LIABILITY

EXCEPT FOR BREACHES OF SECTION 5 (CONFIDENTIALITY) OR INDEMNIFICATION OBLIGATIONS UNDER SECTION
6, IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS
AGREEMENT EXCEED THE TOTAL FEES PAID OR PAYABLE BY CLIENT UNDER THE APPLICABLE STATEMENT OF WORK
DURING THE TWELVE (12) MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM. NEITHER PARTY SHALL
BE LIABLE FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOST
PROFITS, LOSS OF BUSINESS, OR LOSS OF DATA, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES
AND EVEN IF A REMEDY FAILS OF ITS ESSENTIAL PURPOSE. THE FOREGOING LIMITATIONS SHALL APPLY
REGARDLESS OF THE THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, STRICT LIABILITY, OR
OTHERWISE, AND SHALL APPLY EVEN IF ANY LIMITED REMEDY PROVIDED HEREIN FAILS OF ITS ESSENTIAL
PURPOSE.""",
    """8. LIMITATION OF LIABILITY

Service Provider's total liability under this Agreement, whether in contract, tort, or
otherwise, shall not exceed two times (2x) the fees paid by Client in the six (6) months
preceding the claim, except in the case of indemnification obligations for intellectual property
infringement under Section 6, which shall remain uncapped, and except in the case of a breach of
Section 5 (Confidentiality), which shall be capped at an amount equal to five (5) times the
fees paid by Client in the twelve (12) months preceding the claim. Neither party shall be liable
for loss of data, loss of business opportunity, or reputational harm arising from this
Agreement, regardless of whether such damages were foreseeable or whether either party was
advised of the possibility of such damages in advance.""",
]

IP_NEEDLES = [
    """3. INTELLECTUAL PROPERTY OWNERSHIP

All Deliverables specifically created by Service Provider for Client under a Statement of Work
shall be deemed "work made for hire" and shall be owned exclusively by Client upon full payment
therefor. Notwithstanding the foregoing, Service Provider shall retain ownership of any
pre-existing tools, frameworks, methodologies, libraries, or generalized know-how used in
creating the Deliverables ("Service Provider IP"), and hereby grants Client a perpetual,
non-exclusive, royalty-free, worldwide license to use any Service Provider IP embedded in the
Deliverables solely as necessary to use the Deliverables for their intended purpose. Client
shall not reverse engineer, decompile, or attempt to derive the source code of any Service
Provider IP except to the extent such restriction is prohibited by applicable law.""",
    """3. INTELLECTUAL PROPERTY OWNERSHIP

Client shall own all right, title, and interest in the Deliverables, excluding any third-party
or open-source components identified in the applicable Statement of Work, which shall remain
subject to their respective license terms. Service Provider represents that, to its knowledge,
the Deliverables do not infringe the intellectual property rights of any third party, except for
such third-party and open-source components as disclosed to Client in writing prior to delivery,
and Service Provider shall provide Client with a complete list of all such third-party
components, including their applicable license terms, no later than the delivery date of the
affected Deliverables.""",
]


def build_document(company: str, doc_id: int) -> dict:
    indemnification = random.choice(INDEMNIFICATION_NEEDLES)
    liability = random.choice(LIABILITY_NEEDLES)
    ip_clause = random.choice(IP_NEEDLES)

    header = (f"MASTER SERVICES AGREEMENT\n\n"
              f"This Master Services Agreement is entered into between {company} "
              f"(\"Service Provider\") and the client identified in the applicable Statement "
              f"of Work (\"Client\"), effective as of the date of the last signature below.")

    sections = [header, BOILERPLATE_SECTIONS[0], BOILERPLATE_SECTIONS[1], ip_clause,
                BOILERPLATE_SECTIONS[2], BOILERPLATE_SECTIONS[3], indemnification,
                BOILERPLATE_SECTIONS[4], liability, BOILERPLATE_SECTIONS[5],
                BOILERPLATE_SECTIONS[6], BOILERPLATE_SECTIONS[7]]

    return {
        "doc_id": f"msa-{doc_id:03d}",
        "company": company,
        "sections": sections,
        "text": "\n\n".join(sections),
        "indemnification_clause": indemnification,
        "liability_clause": liability,
        "ip_clause": ip_clause,
    }


def chunk_document(doc: dict) -> list[dict]:
    """One section = one chunk, always. Never merges two sections together,
    so a chunk never mixes two unrelated clauses -- that mixing is exactly
    what makes coarse RAG chunks expensive to feed an LLM in the first place."""
    return [
        {"doc_id": doc["doc_id"], "company": doc["company"], "chunk_id": f"{doc['doc_id']}-c{i}", "text": section}
        for i, section in enumerate(doc["sections"])
    ]


def main():
    docs = [build_document(company, i) for i, company in enumerate(COMPANIES)]
    chunks = [chunk for doc in docs for chunk in chunk_document(doc)]

    out_dir = Path(__file__).parent
    (out_dir / "documents.json").write_text(json.dumps(docs, indent=2))
    (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=2))

    print(f"Generated {len(docs)} documents, {len(chunks)} chunks")
    print(f"Avg chunk length: {sum(len(c['text'].split()) for c in chunks) / len(chunks):.0f} words")


if __name__ == "__main__":
    main()
