<?xml version="1.0" encoding="UTF-8"?>
<!-- parse() collects networks from POU bodies and SFC step actions only, so
     a transition body's rungs reach nothing (#7354). -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="SAFE-LD" productVersion="1.0" creationDateTime="2026-08-27"/>
  <contentHeader name="Mutex"/>
  <types/>
  <instances/>
  <pous>
    <pou name="Mutex" pouType="program">
      <interface>
        <outputVars>
          <variable name="A"><type><BOOL/></type></variable>
          <variable name="B"><type><BOOL/></type></variable>
        </outputVars>
      </interface>
      <body>
        <LD>
          <rung localId="1" lineNumber="1"><coil variable="A"/></rung>
        </LD>
      </body>
      <transitions>
        <transition name="T1">
          <body>
            <LD>
              <rung localId="2" lineNumber="2"><coil variable="B"/></rung>
            </LD>
          </body>
        </transition>
      </transitions>
    </pou>
  </pous>
</project>
