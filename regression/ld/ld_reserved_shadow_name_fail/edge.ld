<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-12"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <!-- A declared name that starts with a generated shadow prefix is refused. -->
      <interface>
        <inputVars><variable name="a"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="__edge_prev_a"><type><BOOL/></type><initialValue><simpleValue value="TRUE"/></initialValue></variable><variable name="q"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="a" edge="rising"/><coil variable="q"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
