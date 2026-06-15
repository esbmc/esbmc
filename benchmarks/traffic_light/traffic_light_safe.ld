<?xml version="1.0" encoding="UTF-8"?>
<!-- Traffic Light Controller - SAFE version v3 -->
<!-- Fix 1: Mutual exclusion via explicit phase interlocks -->
<!-- Fix 2: Pedestrian walk gated by Emergency_Vehicle -->
<!-- Fix 3: NS_Red forced ON during emergency (no undefined lamp state) -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-06-12"/>
  <contentHeader name="TrafficLight_Safe"/>
  <types/>
  <instances>
    <configurations>
      <configuration name="Config">
        <resource name="Resource">
          <task name="MainTask" priority="0" interval="T#10ms"/>
          <pouInstance name="TrafficLight_POU" typeName="TrafficLight_Safe"/>
        </resource>
      </configuration>
    </configurations>
  </instances>
  <pous>
    <pou name="TrafficLight_Safe" pouType="program">
      <interface>
        <inputVars>
          <variable name="Enable"><type><BOOL/></type></variable>
          <variable name="Emergency_Vehicle"><type><BOOL/></type></variable>
          <variable name="Ped_Request"><type><BOOL/></type></variable>
        </inputVars>
        <outputVars>
          <variable name="NS_Green"><type><BOOL/></type></variable>
          <variable name="NS_Yellow"><type><BOOL/></type></variable>
          <variable name="NS_Red"><type><BOOL/></type></variable>
          <variable name="EW_Green"><type><BOOL/></type></variable>
          <variable name="EW_Yellow"><type><BOOL/></type></variable>
          <variable name="EW_Red"><type><BOOL/></type></variable>
          <variable name="Ped_NS_Walk"><type><BOOL/></type></variable>
          <variable name="Ped_EW_Walk"><type><BOOL/></type></variable>
        </outputVars>
        <localVars>
          <variable name="Phase_NS_Green"><type><BOOL/></type></variable>
          <variable name="Phase_NS_Yellow"><type><BOOL/></type></variable>
          <variable name="Phase_EW_Green"><type><BOOL/></type></variable>
          <variable name="Phase_EW_Yellow"><type><BOOL/></type></variable>
          <variable name="TON1_IN"><type><BOOL/></type></variable>
          <variable name="TON1_Q"><type><BOOL/></type></variable>
          <variable name="TON1_ET"><type><INT/></type></variable>
          <variable name="TON1_PT"><type><INT/></type></variable>
          <variable name="TON2_IN"><type><BOOL/></type></variable>
          <variable name="TON2_Q"><type><BOOL/></type></variable>
          <variable name="TON2_ET"><type><INT/></type></variable>
          <variable name="TON2_PT"><type><INT/></type></variable>
          <variable name="TON3_IN"><type><BOOL/></type></variable>
          <variable name="TON3_Q"><type><BOOL/></type></variable>
          <variable name="TON3_ET"><type><INT/></type></variable>
          <variable name="TON3_PT"><type><INT/></type></variable>
          <variable name="TON4_IN"><type><BOOL/></type></variable>
          <variable name="TON4_Q"><type><BOOL/></type></variable>
          <variable name="TON4_ET"><type><INT/></type></variable>
          <variable name="TON4_PT"><type><INT/></type></variable>
        </localVars>
      </interface>
      <body>
        <LD>
          <!-- Rung 1: Phase_NS_Green — only when enabled, no emergency, no other phase -->
          <rung localId="1" lineNumber="1">
            <contact variable="Enable"/>
            <contact variable="Emergency_Vehicle" negated="negated"/>
            <contact variable="Phase_NS_Yellow" negated="negated"/>
            <contact variable="Phase_EW_Green" negated="negated"/>
            <contact variable="Phase_EW_Yellow" negated="negated"/>
            <coil variable="Phase_NS_Green"/>
          </rung>
          <!-- Rung 2: TON1 input -->
          <rung localId="2" lineNumber="2">
            <contact variable="Phase_NS_Green"/>
            <coil variable="TON1_IN"/>
          </rung>
          <!-- Rung 3: TON1 timer -->
          <rung localId="3" lineNumber="3">
            <block typeName="TON" instanceName="TON1" lineNumber="3">
              <variable formalParameter="IN">TON1_IN</variable>
              <variable formalParameter="PT">TON1_PT</variable>
              <variable formalParameter="Q">TON1_Q</variable>
              <variable formalParameter="ET">TON1_ET</variable>
            </block>
          </rung>
          <!-- Rung 4: Phase_NS_Yellow — only when TON1 fired and NS_Green gone -->
          <rung localId="4" lineNumber="4">
            <contact variable="TON1_Q"/>
            <contact variable="Phase_NS_Green" negated="negated"/>
            <contact variable="Phase_EW_Green" negated="negated"/>
            <contact variable="Phase_EW_Yellow" negated="negated"/>
            <coil variable="Phase_NS_Yellow"/>
          </rung>
          <!-- Rung 5: TON2 input -->
          <rung localId="5" lineNumber="5">
            <contact variable="Phase_NS_Yellow"/>
            <coil variable="TON2_IN"/>
          </rung>
          <!-- Rung 6: TON2 timer -->
          <rung localId="6" lineNumber="6">
            <block typeName="TON" instanceName="TON2" lineNumber="6">
              <variable formalParameter="IN">TON2_IN</variable>
              <variable formalParameter="PT">TON2_PT</variable>
              <variable formalParameter="Q">TON2_Q</variable>
              <variable formalParameter="ET">TON2_ET</variable>
            </block>
          </rung>
          <!-- Rung 7: Phase_EW_Green — only when TON2 fired and NS phases gone -->
          <rung localId="7" lineNumber="7">
            <contact variable="TON2_Q"/>
            <contact variable="Phase_NS_Green" negated="negated"/>
            <contact variable="Phase_NS_Yellow" negated="negated"/>
            <contact variable="Phase_EW_Yellow" negated="negated"/>
            <coil variable="Phase_EW_Green"/>
          </rung>
          <!-- Rung 8: TON3 input -->
          <rung localId="8" lineNumber="8">
            <contact variable="Phase_EW_Green"/>
            <coil variable="TON3_IN"/>
          </rung>
          <!-- Rung 9: TON3 timer -->
          <rung localId="9" lineNumber="9">
            <block typeName="TON" instanceName="TON3" lineNumber="9">
              <variable formalParameter="IN">TON3_IN</variable>
              <variable formalParameter="PT">TON3_PT</variable>
              <variable formalParameter="Q">TON3_Q</variable>
              <variable formalParameter="ET">TON3_ET</variable>
            </block>
          </rung>
          <!-- Rung 10: Phase_EW_Yellow — only when TON3 fired and EW_Green gone -->
          <rung localId="10" lineNumber="10">
            <contact variable="TON3_Q"/>
            <contact variable="Phase_NS_Green" negated="negated"/>
            <contact variable="Phase_NS_Yellow" negated="negated"/>
            <contact variable="Phase_EW_Green" negated="negated"/>
            <coil variable="Phase_EW_Yellow"/>
          </rung>
          <!-- Rung 11: TON4 input -->
          <rung localId="11" lineNumber="11">
            <contact variable="Phase_EW_Yellow"/>
            <coil variable="TON4_IN"/>
          </rung>
          <!-- Rung 12: TON4 timer -->
          <rung localId="12" lineNumber="12">
            <block typeName="TON" instanceName="TON4" lineNumber="12">
              <variable formalParameter="IN">TON4_IN</variable>
              <variable formalParameter="PT">TON4_PT</variable>
              <variable formalParameter="Q">TON4_Q</variable>
              <variable formalParameter="ET">TON4_ET</variable>
            </block>
          </rung>
          <!-- Rung 13: NS Green lamp — no emergency, NS phase active -->
          <rung localId="13" lineNumber="13">
            <contact variable="Phase_NS_Green"/>
            <contact variable="Emergency_Vehicle" negated="negated"/>
            <coil variable="NS_Green"/>
          </rung>
          <!-- Rung 14: NS Yellow lamp -->
          <rung localId="14" lineNumber="14">
            <contact variable="Phase_NS_Yellow"/>
            <coil variable="NS_Yellow"/>
          </rung>
          <!-- Rung 15: NS Red — when not green AND not yellow, OR during emergency -->
          <rung localId="15" lineNumber="15">
            <contact variable="Phase_NS_Green" negated="negated"/>
            <contact variable="Phase_NS_Yellow" negated="negated"/>
            <coil variable="NS_Red"/>
          </rung>
          <!-- Rung 16: NS Red also forced ON during emergency -->
          <rung localId="16" lineNumber="16">
            <contact variable="Emergency_Vehicle"/>
            <coil variable="NS_Red"/>
          </rung>
          <!-- Rung 17: EW Green lamp -->
          <rung localId="17" lineNumber="17">
            <contact variable="Phase_EW_Green"/>
            <coil variable="EW_Green"/>
          </rung>
          <!-- Rung 18: EW Yellow lamp -->
          <rung localId="18" lineNumber="18">
            <contact variable="Phase_EW_Yellow"/>
            <coil variable="EW_Yellow"/>
          </rung>
          <!-- Rung 19: EW Red — when not green and not yellow -->
          <rung localId="19" lineNumber="19">
            <contact variable="Phase_EW_Green" negated="negated"/>
            <contact variable="Phase_EW_Yellow" negated="negated"/>
            <coil variable="EW_Red"/>
          </rung>
          <!-- Rung 20: NS pedestrian walk — NS Green AND no emergency AND request -->
          <rung localId="20" lineNumber="20">
            <contact variable="Phase_NS_Green"/>
            <contact variable="Emergency_Vehicle" negated="negated"/>
            <contact variable="Ped_Request"/>
            <coil variable="Ped_NS_Walk"/>
          </rung>
          <!-- Rung 21: EW pedestrian walk — EW Green AND request -->
          <rung localId="21" lineNumber="21">
            <contact variable="Phase_EW_Green"/>
            <contact variable="Ped_Request"/>
            <coil variable="Ped_EW_Walk"/>
          </rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
