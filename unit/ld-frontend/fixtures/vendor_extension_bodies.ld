<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="VendorBodies"/>
  <types/>
  <instances/>
  <pous>
    <pou name="VendorBodies" pouType="program">
      <interface>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><coil variable="A"/></rung>
        </LD>
      </body>
      <addData>
        <data name="vendor.widget" handleUnknown="implementation">
          <widget><body><canvas/></body></widget>
        </data>
      </addData>
      <documentation>
        <xhtml xmlns="http://www.w3.org/1999/xhtml"><body><p>Notes.</p></body></xhtml>
      </documentation>
    </pou>
  </pous>
</project>
