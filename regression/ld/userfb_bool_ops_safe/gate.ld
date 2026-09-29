<?xml version="1.0" encoding="UTF-8"?>
<!-- #7579: Boolean operators in a user function block's ST body. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-09-27"/>
  <contentHeader name="GATE"/>
  <types/><instances/>
  <pous>
    <pou name="GATE" pouType="functionBlock">
      <interface>
        <inputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
          <variable name="C"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="AND_OUT"><type><BOOL/></type></variable>
          <variable name="AMP_OUT"><type><BOOL/></type></variable>
          <variable name="OR_OUT"><type><BOOL/></type></variable>
          <variable name="XOR_OUT"><type><BOOL/></type></variable>
          <variable name="NOT_OUT"><type><BOOL/></type></variable>
          <variable name="PREC_OUT"><type><BOOL/></type></variable>
          <variable name="PAREN_OUT"><type><BOOL/></type></variable>
          <variable name="NOTAND_OUT"><type><BOOL/></type></variable>
          <variable name="EQAND_OUT"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body><ST><![CDATA[
AND_OUT := A AND B;
AMP_OUT := A & B;
OR_OUT := A OR B;
XOR_OUT := A XOR B;
NOT_OUT := NOT A;
PREC_OUT := A OR B AND C;
PAREN_OUT := (A OR B) AND C;
NOTAND_OUT := NOT A AND B;
EQAND_OUT := A = B AND C;
]]></ST></body>
    </pou>
    <pou name="MAIN" pouType="program">
      <interface><outputVars><variable name="Q"><type><BOOL/></type></variable></outputVars></interface>
      <body><LD>
        <block localId="10" typeName="GATE" instanceName="g0"/>
        <outVariable localId="11"><expression>Q</expression>
          <connectionPointIn><connection refLocalId="10" formalParameter="AND_OUT"/></connectionPointIn></outVariable>
      </LD></body>
    </pou>
  </pous>
</project>
