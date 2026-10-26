"""Generate a sample technical manual (fictional product) used by the demo, tests and evaluation.

Usage: python scripts/make_sample_manual.py [output.pdf]
"""
import sys
from pathlib import Path

import pymupdf as fitz

PAGES = [
    """PT-200 Industrial Pressure Transmitter
Installation, Operation and Maintenance Manual - Rev. C

1 Overview
The PT-200 is a smart pressure transmitter for measuring gauge and absolute pressure of liquids, gases and steam in process plants. A piezoresistive silicon sensor converts the process pressure into an electrical signal, which the on-board microcontroller linearises and temperature-compensates before driving the analog output.

Typical applications include pump discharge monitoring, hydraulic systems, compressor control and tank level measurement by hydrostatic pressure.

The transmitter housing is made of 316L stainless steel. The wetted diaphragm is available in 316L stainless steel or Hastelloy C-276 for corrosive media such as seawater, chlorides and acids.""",
    """2 Specifications
2.1 Measurement
Measuring range: 0 to 400 bar (gauge), configurable turndown up to 100:1.
Accuracy: +/-0.075 % of full scale, including non-linearity, hysteresis and repeatability.
Long-term stability: +/-0.1 % of upper range limit per 5 years.
Overpressure limit: 2 times the upper range limit. Burst pressure: 4 times the upper range limit.

2.2 Electrical
Output signal: 4-20 mA two-wire with HART 7 digital communication.
Supply voltage: 10.5 to 42 VDC at the transmitter terminals.
Response time: less than 50 ms.

2.3 Environmental
Operating temperature: -40 to 85 degC (ambient). Process temperature: -40 to 120 degC.
Ingress protection: IP67 according to IEC 60529.
Process connection: 1/2 inch NPT male; G 1/2 available on request.""",
    """3 Installation
3.1 Mounting
Mount the transmitter vertically with the process connection facing down, so that condensate drains away from the sensor. For steam service, install a siphon or condensate loop to keep the diaphragm below 120 degC.

Tighten the process connection to a torque of 25 Nm using a wrench on the hexagon of the process connection only. Never turn the transmitter by its housing, as this can damage the internal wiring.

3.2 Electrical connection
Use shielded twisted-pair cable with a conductor cross-section between 0.5 and 2.5 mm2. Ground the cable shield at the control-room end only to avoid ground loops. The maximum loop resistance is (supply voltage - 10.5 V) / 0.022 A. For a 24 V supply this gives approximately 614 ohm.""",
    """4 Calibration
The PT-200 is factory calibrated. Field calibration is recommended every 12 months, or after any overpressure event, to maintain the stated accuracy.

4.1 Zero adjustment
Vent the transmitter to atmosphere so that the applied pressure is zero. Press and hold the ZERO push-button for 3 seconds until the display shows "ZERO OK". The output must then read 4.00 mA +/- 0.01 mA.

4.2 Span adjustment
Apply the upper range value with a reference pressure calibrator whose accuracy is at least three times better than the transmitter. Press and hold the SPAN push-button for 3 seconds until the display shows "SPAN OK". The output must read 20.00 mA +/- 0.01 mA.

Record the as-found and as-left values in the calibration log.""",
    """5 Troubleshooting
The display and HART diagnostics report the following error codes:
E01 - Sensor fault: the sensor bridge is open or shorted. Return the unit for repair.
E02 - Over-range: the applied pressure exceeds the upper range limit. Check the process and the selected range.
E03 - Supply voltage low: terminal voltage is below 10.5 VDC. Check the power supply and the loop resistance.
E04 - Memory checksum error: configuration data is corrupt. Restore factory settings via HART and re-calibrate.

If the output is fixed at 3.6 mA, the transmitter has detected an internal failure and has driven the output to the low alarm level defined by NAMUR NE 43. A fixed output of 21.0 mA indicates the high alarm level.""",
    """6 Maintenance
The PT-200 has no user-serviceable parts. Inspect the housing seals and cable gland every 6 months for damage or moisture ingress. Clean the diaphragm only with a soft brush and a cleaning agent compatible with the diaphragm material; never use sharp tools.

7 Safety and Certifications
The PT-200 is certified for use in hazardous areas: ATEX and IECEx Ex ia IIC T4 Ga (intrinsically safe). The device is suitable for safety instrumented functions up to SIL 2 according to IEC 61508.

Before removing the transmitter from the process, isolate and depressurise the line. Disconnect power before opening the housing in a hazardous area.

Warranty: 24 months from the date of shipment.""",
]


def main(output: str = "sample_docs/PT-200_Pressure_Transmitter_Manual.pdf") -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in PAGES:
        page = doc.new_page()  # A4-ish default page
        page.insert_textbox(fitz.Rect(56, 56, 540, 790), text, fontsize=10.5, fontname="helv")
    doc.save(output)
    print(f"Wrote {output} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main(*sys.argv[1:])
