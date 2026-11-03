<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-10-12"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <!-- A counter input that only a block pin reads is a free input, so the counter can count. -->
      <interface>
        <inputVars></inputVars>
        <localVars><variable name="Pulse"><type><BOOL/></type></variable><variable name="Never"><type><BOOL/></type></variable><variable name="Done"><type><BOOL/></type></variable><variable name="P"><type><INT/></type><initialValue><simpleValue value="1"/></initialValue></variable><variable name="CV"><type><INT/></type><initialValue><simpleValue value="0"/></initialValue></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><block typeName="CTU" instanceName="c"><variable formalParameter="CU">Pulse</variable><variable formalParameter="R">Never</variable><variable formalParameter="PV">P</variable><variable formalParameter="Q">Done</variable><variable formalParameter="CV">CV</variable></block></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
