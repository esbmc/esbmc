<?xml version="1.0" encoding="UTF-8"?>
<!-- <ladderDiagram> is the schema's second spelling of an LD body. The body
     whitelist has to accept it: rejecting it would drop live rungs, and the
     invariant fails if either rung goes missing (#7354). -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="Alias"/>
  <types/>
  <instances/>
  <pous>
    <pou name="Alias" pouType="program">
      <interface>
        <inputVars>
          <variable name="Enable"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <ladderDiagram>
          <rung localId="1" lineNumber="1">
            <contact variable="Enable"/>
            <coil variable="A"/>
          </rung>
          <rung localId="2" lineNumber="2">
            <contact variable="Enable" negated="true"/>
            <coil variable="B"/>
          </rung>
        </ladderDiagram>
      </body>
    </pou>
  </pous>
</project>
