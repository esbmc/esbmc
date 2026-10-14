<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="SfcBody"/>
  <types/>
  <instances/>
  <pous>
    <pou name="SfcBody" pouType="program">
      <interface>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <SFC>
          <step localId="1" name="Init" initialStep="true"/>
        </SFC>
      </body>
      <actions>
        <action name="Drive">
          <body>
            <LD>
              <rung localId="1"><coil variable="A"/></rung>
            </LD>
          </body>
        </action>
      </actions>
    </pou>
  </pous>
</project>
