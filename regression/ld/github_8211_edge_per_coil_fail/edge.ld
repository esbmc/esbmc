<?xml version="1.0" encoding="UTF-8"?>
<!-- #8211: each coil of a rung re-reads an edge contact, as Beremiz emits one R_TRIG per coil. -->
<project xmlns="http://www.plcopen.org/xml/tc6_0200">
  <fileHeader companyName="ESBMC" productVersion="1.0" creationDateTime="2026-10-20"/>
  <contentHeader name="P"/>
  <types/><instances/>
  <pous>
    <pou name="P" pouType="program">
      <interface>
        <inputVars><variable name="go"><type><BOOL/></type></variable></inputVars>
        <localVars><variable name="v"><type><BOOL/></type></variable><variable name="hit"><type><BOOL/></type></variable></localVars>
      </interface>
      <body>
        <LD>
          <rung localId="1"><contact variable="go"/><coil variable="v"/></rung>
          <rung localId="2"><contact variable="v" edge="rising"/><coil variable="v" kind="reset"/><coil variable="hit"/></rung>
        </LD>
      </body>
    </pou>
  </pous>
</project>
