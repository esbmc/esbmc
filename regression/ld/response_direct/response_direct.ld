<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-10"/>
  <contentHeader name="RespSafe"/>
  <types/>
  <instances><configurations><configuration name="Config"><resource name="R">
    <task name="MainTask" priority="0" interval="T#10ms"/>
    <pouInstance name="P" typeName="RespSafe"/>
  </resource></configuration></configurations></instances>
  <pous><pou name="RespSafe" pouType="program"><interface>
    <inputVars><variable name="Start"><type><BOOL/></type></variable></inputVars>
    <outputVars><variable name="Out"><type><BOOL/></type></variable></outputVars>
  </interface><body><LD>
    <rung localId="1" lineNumber="1">
      <contact variable="Start"/>
      <coil variable="Out"/>
    </rung>
  </LD></body></pou></pous>
</project>
